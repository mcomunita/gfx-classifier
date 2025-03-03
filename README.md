# **Guitar Effects Recognition and Parameter Estimation with Convolutional Neural Networks**

![](img/pedals.jpg)

## **Project**

Despite the popularity of guitar effects, there is very little existing research on classification and parameters  estimation  of specific  plugins  or  effect  units  from  guitar  recordings.   In this research project, convolutional neural networks were used for classification and parameter estimation for 13 overdrive, distortion and fuzz guitar effects. A novel dataset of processed electric guitar samples was assembled, with four sub-datasets consisting of monophonic or polyphonic samples and discrete or continuous settings values, for a total of about 305 hours of processed samples.  Results were compared for networks trained and tested on the same or on a different sub-dataset. We found that discrete datasets could lead to equally high performance as continuous ones, whilst being easier to design, analyse and modify. Classification accuracy was above 80%, with confusion matrices reflecting similarities in the effects timbre and circuits design. With parameters values between 0.0 and 1.0, the mean absolute error is in most cases below 0.05, while the root mean square error is below 0.1 in all cases but one.

## **Paper**
[https://www.aes.org/e-lib/browse.cfm?elib=21124](https://www.aes.org/e-lib/browse.cfm?elib=21124)

## **Dataset**
[https://doi.org/10.5281/zenodo.4296040](https://doi.org/10.5281/zenodo.4296040)

[https://doi.org/10.5281/zenodo.4298000](https://doi.org/10.5281/zenodo.4298000)

[https://doi.org/10.5281/zenodo.4298017](https://doi.org/10.5281/zenodo.4298017)

[https://doi.org/10.5281/zenodo.4298025](https://doi.org/10.5281/zenodo.4298025)


## **Models and Results**

[https://github.com/mcomunita/gfx_classifier_models_and_results](https://github.com/mcomunita/gfx_classifier_models_and_results)

## **Extended results**
[https://mcomunita.github.io/gfx_classifier_page](https://mcomunita.github.io/gfx-classifier_page)

## **Cite**
```
@article{comunità2021guitar,
            title={Guitar Effects Recognition and Parameter Estimation with Convolutional Neural Networks},
            author={Comunità, Marco and  Stowell, Dan and  Reiss, Joshua D.},
            journal={Journal of the Audio Engineering Society},
            year={2021},
            volume={69},
            number={7/8},
            pages={594-604},
            doi={}, 
            month={July}
}
```
