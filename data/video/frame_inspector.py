import os, glob

inspect_dir = 'face_input'
inspect_range = (0,1800)
valid_frame_path = 'valid_frame.txt'

def check_frame(idx,part,dir=inspect_dir):
    path = dir + "/frame_%d_%02d.jpg"%(idx,part)
    if(not os.path.exists(path)): return False
    return True

for i in range(inspect_range[0],inspect_range[1]):
    valid = True
    print('processing frame %s'%i)
    for j in range(1,76):
        if(check_frame(i,j)==False):
            path = inspect_dir + "/frame_%d_*.jpg"% i
            for file in glob.glob(path):
                os.remove(file)
            valid = False
            print('frame %s is not valid'%i)
            break
    if valid:
        with open(valid_frame_path,'a') as f:
            frame_name = "frame_%d"%i
            f.write(frame_name+'\n')



