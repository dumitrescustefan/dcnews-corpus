import os
import subprocess


def main():
    for max_features_item in range(60000, 220000, 20000):
        for ngram_range_max in range(1, 5):
            for slinear_tf in [False]:
                os.system('python explore.py '
                          '--max-features {} '
                          '--ngram-range-max {} '
                          '--sublinear-tf {}'.format(max_features_item, ngram_range_max, slinear_tf))

        # process = subprocess.Popen(['python', 'explore.py', '--max-features', str(max_features_item)],
        #                            stdout=subprocess.PIPE,
        #                            universal_newlines=True)
        #
        # while True:
        #     output = process.stdout.readline()
        #     print(output.strip())
        #     # Do something else
        #     return_code = process.poll()
        #     if return_code is not None:
        #         print('RETURN CODE', return_code)
        #         # Process has finished, read rest of the output
        #         for output in process.stdout.readlines():
        #             print(output.strip())
        #         break


main()
