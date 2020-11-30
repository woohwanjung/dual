import json
import os
use_line_profiler = False
#kernprof -l -v script_to_profile.py

DEBUG_NOSHUFFLE = False
HALF_PRECISION = False
DEBUG_LOGGING_MU = True
DEBUG_NOSAVE = False
WRITER_DIR = "../tb_log"


EXT_DIR = "../result"
WRITER_DIR_DUAL = f"{WRITER_DIR}/log_dual"


EXTRACTION_DIR = f"{EXT_DIR}/extracted"
EXTRACTION_FOR_TEST_DIR = f"{EXT_DIR}/extracted_test"
TEST_RESULT_DIR = f"{EXT_DIR}/test_result"
FIGURE_DIR = "fig_result"

EVAL_DIR = f"../eval"
EVAL_RES_DIR = f"{EVAL_DIR}/res"
EVAL_REF_DIR = f"{EVAL_DIR}/ref"

def conditional_profiler(func):
    if use_line_profiler:
        return profile(func)
    return func



if __name__=="__main__":
    dirs = [
        WRITER_DIR,
        WRITER_DIR_DUAL,
        EXT_DIR,
        EXTRACTION_DIR,
        EXTRACTION_FOR_TEST_DIR,
        TEST_RESULT_DIR,
        FIGURE_DIR,
        EVAL_DIR,
        EVAL_RES_DIR,
        EVAL_REF_DIR,
    ]
    for dirpath in dirs:
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
            print(dirpath)
