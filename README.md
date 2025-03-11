# RangeNet Semantic Segmentation

## Source of the model

	  Model picked up from 'https://github.com/PRBonn/lidar-bonnetal/tree/master'

---

## Description of the model

	> Model     : RangeNet
    > Backbone  : Squeezeseg
	> Input size: [1, 5, 64, 2048]

---

## Framework and version

    AIMET   : torch-gpu-1.24.0
    offset  : 11
    pytorch : 1.9.1+cu111
    python  : 3.8


## Trained on dataset(s)

Kitti Pretrained

## Changes done in the Code 

In `aimet-rangenet/src/common/laserscan.py (reset function)` set all the **np.float to np.float 32**

```python

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.int32)          # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=np.float32)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=np.float32)              # [H,W,3] color

```
---

In `aimet-rangenet/src/backbones/squeezeseg.py` 

```python

  ##MCW
  
  """ Old Code 
  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os
  
  

  def forward(self, x):
    # filter input
    x = x[:, self.input_idxs]

    # run cnn
    # store for skip connections
    skips = {}
    os = 1

    # encoder
    skip_in = self.conv1b(x)
    x = self.conv1a(x)
    # first skip done manually
    skips[1] = skip_in.detach()
    os *= 2

    x, skips, os = self.run_layer(x, self.fire23, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.fire45, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.fire6789, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)

    return x, skips
    
    """

    """ New Code 
    
    Replace Run Layer with run layer 1 and run layer 2 
    
    """
    
    def run_layer1(self, x, layer, skips, os):
      y = layer(x)
      skips[os] = x.detach()
      os *= 2
      x = y
      return x, skips, os
  
    def run_layer2(self, x, layer, skips, os):
      y = layer(x)
      x = y
      return x, skips, os
  
  

  def forward(self, x):
    # filter input
    x = x[:, self.input_idxs]

    # run cnn
    # store for skip connections
    skips = {}
    os = 1

    # encoder
    skip_in = self.conv1b(x)
    x = self.conv1a(x)
    # first skip done manually
    skips[1] = skip_in.detach()
    os *= 2

    x, skips, os = self.run_layer1(x, self.fire23, skips, os)
    x, skips, os = self.run_layer2(x, self.dropout, skips, os)
    x, skips, os = self.run_layer1(x, self.fire45, skips, os)
    x, skips, os = self.run_layer2(x, self.dropout, skips, os)
    x, skips, os = self.run_layer1(x, self.fire6789, skips, os)
    x, skips, os = self.run_layer2(x, self.dropout, skips, os)

    return x, skips

```

---

In `aimet-rangenet/src/decoders/squeezeseg.py` 

```python

# MCW
  
  """ Old Code
  def run_layer(self, x, layer, skips, os):
    feats = layer(x)  # up
    if feats.shape[-1] > x.shape[-1]:
      os //= 2  # match skip
      feats = feats + skips[os].detach()  # add skip
    x = feats
    return x, skips, os
    
    """
    
    #New Code -> Removed if Condition
    def run_layer(self, x, layer, skips, os):
      feats = layer(x)  # up
      os //= 2  # match skip
      feats = feats + skips[os].detach()  # add skip
      x = feats
      return x, skips, os

```

## Result 

  - RangeNet Squeezeseg
    - FP32 
        - Acc avg: 0.761
        - IoU avg: 0.305

    - Quantized BN CLE PTQ
        - Acc avg 0.419
        - IoU avg 0.152

    - Quantized QAT (2 Epochs)
        - Acc avg 0.711
        - IoU avg 0.267

        
---

