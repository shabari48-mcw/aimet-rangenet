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
## Result 

  - RangeNet Squeezeseg
    - FP32 
        - Acc avg: 0.761
        - IoU avg: 0.305
        
---

