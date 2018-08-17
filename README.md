[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

## Recurrent Neural Network with Timestep and LSTM Cell for Protein Secondary Structure Prediction

Built without use of any framework besides tensorflow's low level API (see tf.matmul,tf.add, etc).
Accuracy is currently at 81%
Built by @neelsankaran and I

## Data,Labels,and Loss
Input Data: Protein Primary Structure | Shape: (5600(#examples),700(#amino acids in sequence),22(#features in embeding))
Labels: Protein Secondary Structure | Shape: (5600(#examples),700(#amino acids in sequence),10(#features in embeding))
Loss Function: Cross Entropy

## Diagram w/ Explanation
         
         
                             Y^(0)      | LSTM Updates A With Previous Relevant Data |
                               ^                              ^
                               |                              |             Next Y^
                               |                              |                ^
             ----------------------------------------         |                |
             |                                      |         |         -----------------
             |                                      |         |         |               |
             | Y^ = tanH(conc*Wy'+By)               |         |         |               |
    A(0) ->  | nextA = tanH((In*Wx'+Bx)+(A*Wa'+Ba)) | ----> nextA ----> | NextTimeStep  |....N=700 -> Calc Cost With All Y^'s |
             |                                      |                   |               |                                     |
             |                                      |                   |               |                                     |
             |                                      |                   |               |                                     |
             ----------------------------------------                   -----------------                                     |
                                ^                                               ^                                             |
                                |                                               |                                             |
                                |                                           Next Input                                        |
                           Input Data(0)                                                                                      |                                                                                                                               |
                                                 Gradient From Cost Flows Back                                                |
         <---------------------------------------------------------------------------------------------------------------------
