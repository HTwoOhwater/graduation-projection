1.Objective: predict depth for images and add fog
2.Procedures:
2.1 change settings in test.sh     --image_path  source images path
                                   --save_path   result images path
                                   --model_name  pretrained models in models/
                                   --add_fog_method  {mean, mean+min+max, mean+min+max+th}
                                        mean: mean filtering
                                        mean+min+max : three filtering methods for closer near and farther in the distance
                                        mean+min+max+th: three filtering methods + threshold to change the deepest position to 85%
2.2 run test.sh
3.source code: https://github.com/nianticlabs/monodepth2