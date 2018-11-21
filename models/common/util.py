import subprocess


def get_num_total_lines(p):
    # result = subprocess.run(['wc', '-l', p], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if not isinstance(p, str):
        p = str(p)

    result = subprocess.check_output(['wc', '-l', p])
    if isinstance(result, bytes):
        result = result.decode('utf-8')

    return int(result.strip().split(' ')[0])
