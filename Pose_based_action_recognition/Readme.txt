Before running the project yu have to download keras model for openpose. File name: 'model.h5' and put it in 'model/keras' forlder.
Here is the link: https://drive.google.com/drive/folders/1KCmEngNblVignMcQJp4MdwUMUE_VRWou?usp=sharing

report.pdf - report of the project

Ready-made examples of videos for testing are in the 'videos/Test' folder, and their results in the 
'videos/outputs_LSTM' and 'videos/outputs_Transformer' folder.

To test the program with your video:
1) Transfer the video to folder 'videos/Test'. Required.
2) Run the file 'demo_video.py', specifying the '--video' argument and '--classifier'. 
You only need to specify the video name, NOT THE PATH. 
Classifier must either 'LSTM' or 'Transformer'.
3) The result of the video will be saved in the 'videos/outputs' folder,
and its name will match the name of your video.

Example of console command:
python demo_video.py --video 'sample_video.mp4' -- classifier 'LSTM'
python demo_video.py --video 'sample_video.mp4' -- classifier 'Transformer'
