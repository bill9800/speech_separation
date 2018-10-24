import sys
sys.path.append("../lib")
import AVHandler as avh


def simple_audio(loc,name,link1,link2,start,end):
    # A function to get 2 audio mix together
    # loc   | the location for file to store
    # name  | name for the wav mix file
    # link1 | link to download the file1
    # link2 | link to download the file2
    # start | audio starting time
    # end   | audio end time
    name1 = '%s_1' % name
    name2 = '%s_2' % name
    avh.download(loc,name1,link1)
    avh.download(loc,name2,link2)
    avh.mix(loc,name,name1,name2,start,end,trim_clean=False)

avh.mkdir('simple_audio')

# simple audio mix1
# 2 man mixture 3 min

link1_1 = "https://www.youtube.com/watch?v=4q1dgn_C0AU"
link1_2 = "https://www.youtube.com/watch?v=DCS6t6NUAGQ"
simple_audio('simple_audio','simple_audio1',link1_1,link1_2,360,540)

# simple audio mix2
# 2 women mixture 3 min

link2_1 = "https://www.youtube.com/watch?v=cQqmMVOsriI"
link2_2 = "https://www.youtube.com/watch?v=MwvctN3Uejg"
simple_audio('simple_audio','simple_audio2',link2_1,link2_2,360,540)

# simple audio mix2
# 1 women and 1 man mixture 3 min
link3_1 = "https://www.youtube.com/watch?v=gh5VhaicC6g"
link3_2 = "https://www.youtube.com/watch?v=w-HYZv6HzAs"
simple_audio('simple_audio','simple_audio3',link3_1,link3_2,360,540)







