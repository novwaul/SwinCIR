# SwinCIR: SwinIR with multi-resolution connections
<img width="1061" alt="스크린샷 2022-12-20 오후 10 46 49" src="https://user-images.githubusercontent.com/53179332/208681748-4e043bba-c207-4c56-9f24-fc89d4c45577.png">


## 4x Result

### PSNR
|Dataset|Bicubic|SwinCIR|
|:---:|:---:|:---:|
|Set5|28.648|32.460|
|Set14|26.230|28.793|
|Urban100|23.220|26.329|

### Set14
| Bicubic | SwinCIR | GT |
|:---:|:---:|:---:|
|<img width="264" alt="image" src="https://user-images.githubusercontent.com/53179332/204194615-cb470e05-fd2a-46cc-aae3-0ac42f0ec7b5.png">|<img width="264" alt="image" src="https://user-images.githubusercontent.com/53179332/208682574-822d8690-01fe-42de-a49c-439d4e428b29.png">|<img width="264" alt="image" src="https://user-images.githubusercontent.com/53179332/204194543-d00bd079-1348-4f96-b929-c7a7a52fd042.png">|




## Train Setting
|Item|Setting|
|:---:|:---:|
|Train Data|DIV2K, Flickr2K|
|Preprocess|[-1,1] Normalization |
|Random Transforms|Crop {DIV2K(64x64), Flickr2K(64x64)}, Rotation {90 degree} |
|Validation Data|DIV2K|
|Test Data| Set5, Set14, Urban100|
|Scale| 4x |
|Optimizer|Adam|
|Learning Rate|2e-4|
|Scheduler|Cosine Annealing with last LR of 1e-6|
|Trained Iterations|5e5|
|Loss|L1|
|Batch|4 {2 for each GPU, total 2 GPUs are utilized} |
