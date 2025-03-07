#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.backends.cudnn as cudnn
import imp
import time
import os
import sys
import numpy as np
import torch

from modules.ioueval import iouEval
from common.laserscan import SemLaserScan
from modules.segmentator import *
from postproc.KNN import KNN


from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_common.defs import QuantScheme
from utils.val import validate_model

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,config):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.config=config
    self.input_shape=config['input_shape']

    

    # get the data
    parserModule = imp.load_source("parserModule",
                                   'src/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.dummy_input=torch.randn(self.input_shape,device=self.device)
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):

    # do valid set
    self.infer_subset(loader=self.parser.get_valid_set(),
                      to_orig_fn=self.parser.to_original)

    print('Evaluating the Model........')
    
    self.eval()

    return

  def infer_subset(self, loader, to_orig_fn):
    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        # compute output
        proj_output = self.model(proj_in, proj_mask)
        proj_argmax = proj_output[0].argmax(dim=0)

        if self.post:
          # knn postproc
          unproj_argmax = self.post(proj_range,
                                    unproj_range,
                                    proj_argmax,
                                    p_x,
                                    p_y)
        else:
          # put in original pointcloud using indexes
          unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
          torch.cuda.synchronize()

        print("Infered seq", path_seq, "scan", path_name,
              "in", time.time() - end, "sec")
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)

  def quant(self):
    
      self.model=validate_model(self.model,self.dummy_input)

      if len(self.config['methods']["ptq"])>0:
          print("Manual PTQ..........\n\n")
      
      
      #Apply PTQ techniques based on flags    
      
      if "bn" in self.config['methods']["ptq"]:
          self.apply_bnfold() 
          
      if "cle" in self.config['methods']["ptq"]:
          self.apply_cle()
          
      if "ada" in self.config['methods']["ptq"]:
          self.apply_adaround(self.config)
          
          
      print("\nQuantization Sim\n\n")
      
      scheme=QuantScheme.post_training_tf_enhanced if self.config["quantization_configuration"]["quant_scheme"]== "post_training_tf_enhanced" else QuantScheme.training_range_learning_with_tf_enhanced_init


      kwargs = {
              "quant_scheme":scheme ,
              "default_param_bw": self.config['quantization_configuration']['param_bw'],
              "default_output_bw": self.config['quantization_configuration']['output_bw'], 
              "dummy_input": self.dummy_input,
          }
    
      
      sim=QuantizationSimModel(self.model,**kwargs)
          
      if "ada" in self.config['methods']["ptq"]:
          print("\n\n <-------------Set and Freeze the Param encodings for AdaRound --------->\n\n")
          sim.set_and_freeze_param_encodings(encoding_path=os.path.join(self.config['exports_path'],'adaround.encodings'))
      
      sim.compute_encodings(self.infer_subset,self.parser.get_valid_set())  
                                  
      if len(self.config['methods']["ptq"])>0: 
          
          print("\n\n <-------------Evaluating the PTQ model--------->\n\n")
          
          
          self.model=sim.model
          
          self.infer()
           
              
          print("\n\n <-------------Exporting the PTQ model--------->\n\n")
      
          
          sim.export(path=self.config['exports_path'],filename_prefix=self.config['export-name'],dummy_input=self.dummy_input.cpu())
          
     
          
      
      
      #Apply Quantization Aware Training , Evaluate and Save the model
      if self.config['methods']['qat']:
        pass
      
       #Apply AutoQuant 
        
      if self.config['methods']["autoquant"]:
            pass
          
          # print("\n\nAutoQuant....\n\n")
          # dataset,data_loader=load_unlabelled_data(config,config['quantconfg']['calibration_class'],config['quantconfg']['calibration_images'],32)
          # dummy_input=dummy_input.cuda()
          # model.to("cuda")
          # auto_quant = AutoQuant(model,
          #               dummy_input=dummy_input,
          #               data_loader=data_loader,
          #               eval_callback=auto_callback)
          
          # model, optimized_accuracy, encoding_path = auto_quant.optimize(allowed_accuracy_drop=config['allowed_accuracy_drop'])
          
          # print(f"\nTop1 Quantized Accuracy (after optimization):  {optimized_accuracy}\n")
        
  def eval(self):

      DATA=self.DATA

      # get number of interest classes, and the label mappings
      class_strings = DATA["labels"]
      class_remap = DATA["learning_map"]
      class_inv_remap = DATA["learning_map_inv"]
      class_ignore = DATA["learning_ignore"]
      nr_classes = len(class_inv_remap)

      # make lookup table for mapping
      maxkey = 0
      for key, data in class_remap.items():
        if key > maxkey:
          maxkey = key
      # +100 hack making lut bigger just in case there are unknown labels
      remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
      for key, data in class_remap.items():
        try:
          remap_lut[key] = data
        except IndexError:
          print("Wrong key ", key)
      # print(remap_lut)

      # create evaluator
      ignore = []
      for cl, ign in class_ignore.items():
        if ign:
          x_cl = int(cl)
          ignore.append(x_cl)
          print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

      # create evaluator
      device = torch.device("cpu")
      evaluator = iouEval(nr_classes, device, ignore)
      evaluator.reset()

      # get test set
      test_sequences = DATA["split"]["valid"]

      # get scan paths
      scan_names = []
      for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        scan_paths = os.path.join(self.datadir, "sequences",
                                  str(sequence), "velodyne")
        # populate the scan names
        seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
        seq_scan_names.sort()
        scan_names.extend(seq_scan_names)
      # print(scan_names)

      # get label paths
      label_names = []
      for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(self.datadir, "sequences",
                                  str(sequence), "labels")
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if ".label" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)
      # print(label_names)

      # get predictions paths
      pred_names = []
      for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        pred_paths = os.path.join(self.logdir, "sequences",
                                  sequence, "predictions")
        # populate the label names
        seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)
      # print(pred_names)

      # check that I have the same number of files
      # print("labels: ", len(label_names))
      # print("predictions: ", len(pred_names))
      
      assert(len(label_names) == len(scan_names) and
            len(label_names) == len(pred_names))

      print("Evaluating sequences: ")
      # open each file, get the tensor, and make the iou comparison
      for scan_file, label_file, pred_file in zip(scan_names, label_names, pred_names):
        print("evaluating label ", label_file, "with", pred_file)
        # open label
        label = SemLaserScan(project=False)
        label.open_scan(scan_file)
        label.open_label(label_file)
        u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
        

        # open prediction
        pred = SemLaserScan(project=False)
        pred.open_scan(scan_file)
        pred.open_label(pred_file)
        u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format
        
        # add single scan to evaluation
        evaluator.addBatch(u_pred_sem, u_label_sem)

      # when I am done, print the evaluation
      m_accuracy = evaluator.getacc()
      m_jaccard, class_jaccard = evaluator.getIoU()

      print('Validation set:\n'
            'Acc avg {m_accuracy:.3f}\n'
            'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,
                                            m_jaccard=m_jaccard))
      # print also classwise
      for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
          print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
              i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

      # print for spreadsheet
      print("*" * 80)
      print("below can be copied straight for paper table")
      for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
          sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
          sys.stdout.write(",")
      sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
      sys.stdout.write(",")
      sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
      sys.stdout.write('\n')
      sys.stdout.flush()



  def apply_adaround(self,config):
          
          print("\AdaRound......\n\n")
          
          
          qconfig=config['quantconfg']
          
          #Set the Parameters for performing ADA Round
          
          params = AdaroundParameters(data_loader=self.parser.get_valid_set(), num_batches=1,default_num_iterations=1)
          self.model = Adaround.apply_adaround(self.model, self.dummy_input, params,
                                              path=config['exports_path'], filename_prefix='adaround', default_param_bw=config['quantization_configuration']['param_bw'],
                                              default_quant_scheme=QuantScheme.post_training_tf_enhanced)
          


      
  def apply_cle(self,use_cuda: bool = True) :
      
      print("\nCLE......\n\n")
  
      if use_cuda:
          self.model=self.model.cuda()

      equalize_model(self.model,self.input_shape)
      
      
      
  def apply_bnfold(self,use_cuda: bool = True):
      
      print("\nBatchNormFolding......\n\n")
      
      if use_cuda:
          self.model=self.model.cuda()
      
      _ = fold_all_batch_norms(self.model, self.input_shape)              