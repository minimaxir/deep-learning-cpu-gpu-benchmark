import os
import sys

platform_type = sys.argv[1]   # cpu08, cpu16, cpu32, cpu64, gpu

if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('logs/{}'.format(platform_type)):
    os.makedirs('logs/{}'.format(platform_type))

test_files = [f for f in os.listdir("test_files") if f.endswith('.py')]
test_files.remove('CustomCallback.py')
docker_type = 'nvidia-' if 'gpu' in platform_type else ''
docker_cmd = "sudo {}docker run -it --rm -v".format(docker_type) + \
    "$(pwd)/:/keras --name keras"

for test_file in test_files:
    tag = ':cpu' if 'cpu' in platform_type else ''

    statement = docker_cmd + \
        " -e KERAS_BACKEND='tensorflow' minimaxir/keras-cntk{} ".format(
            tag) + \
        "python3 test_files/{} {}".format(test_file, platform_type)

    print(statement)

    os.system(statement)
