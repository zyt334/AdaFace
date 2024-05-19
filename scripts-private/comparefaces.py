import os
import argparse
import time
from eval_utils import compare_face_folders_fast, set_tf_gpu, video_frame_extraction

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path1", type=str, help="path1")
    parser.add_argument("path2", type=str, nargs='?', default=None, help="path2")
    parser.add_argument("--self", dest="self_comp", action="store_true", help="self comparison")
    parser.add_argument("--pair", dest="folderpair_comp", action="store_true", help="subfolder pairwise comparison")
    parser.add_argument("--gpu", dest='gpu_id', type=int, default=0, help="specify a gpu to use")
    parser.add_argument("--face_engine", dest='face_engine', type=str, default='deepface', 
                        choices=['deepface', 'insightface'],
                        help="face engine to use for comparison")
    parser.add_argument("--video", dest="video_comp", action="store_true", default=False, help="video image comparison")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    set_tf_gpu(args.gpu_id)

    skip_subjs = [ 'lilbub', 'jiffpom', 'princessmonstertruck' ]

    begin = time.time()
    
    # currently does not work if both path1 and path2 contain videos
    # since the function writes to a common file and will overwrite each other
    if args.video_comp:
        if not args.folderpair_comp:
            if video_frame_extraction(args.path1):
                #set args.path1 to new dir containing extracted images
                args.path1 = video_frame_extraction(args.path1)
            if video_frame_extraction(args.path2):
                #if args.path2 contains videos instead
                args.path2 = video_frame_extraction(args.path2)

    if not args.folderpair_comp:
        if args.self_comp:
            subdirs = sorted(os.listdir(args.path1))
            subdir_count = 0
            for subdir in subdirs:
                subdir_path = os.path.join(args.path1, subdir)
                if os.path.isdir(subdir_path) == False:
                    continue

                print("Self comparing %s:" %(subdir_path))
                compare_face_folders_fast(subdir_path, subdir_path, face_engine=args.face_engine)
                subdir_count += 1
            if subdir_count == 0:
                print("Self comparing %s:" %(args.path1))
                compare_face_folders_fast(args.path1, args.path1, face_engine=args.face_engine)
        else:            
            compare_face_folders_fast(args.path1, args.path2, face_engine=args.face_engine)
    else:
        subdirs = sorted(os.listdir(args.path1))
        for subdir in subdirs:
            if subdir in skip_subjs:
                continue
            subdir_path1 = os.path.join(args.path1, subdir)
            subdir_path2 = os.path.join(args.path2, subdir)
            if args.video_comp:
                if video_frame_extraction(subdir_path1):
                    #set args.path1 to new dir containing extracted images
                    subdir_path1 = video_frame_extraction(subdir_path1)
                if video_frame_extraction(subdir_path2):
                    #if args.path2 contains videos instead
                    subdir_path2 = video_frame_extraction(subdir_path2)
            if os.path.isdir(subdir_path1) == False or os.path.isdir(subdir_path2) == False:
                continue
            print("Pair comparing %s vs %s:" %(subdir_path1, subdir_path2))
            compare_face_folders_fast(subdir_path1, subdir_path2, face_engine=args.face_engine)     
        
    end = time.time()
    print("Time elapsed: %.2f seconds" %(end - begin))
            print("Pair comparing %s vs %s:" %(subdir_path1, subdir_path2))
            compare_face_folders_fast(subdir_path1, subdir_path2, face_engine=args.face_engine)     
        
    end = time.time()
    print("Time elapsed: %.2f seconds" %(end - begin))
