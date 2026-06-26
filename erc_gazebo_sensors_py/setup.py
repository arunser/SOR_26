from setuptools import find_packages, setup

package_name = 'erc_gazebo_sensors_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'chase_the_ball = erc_gazebo_sensors_py.chase_the_ball:main',
            'yolo_detection_node = erc_gazebo_sensors_py.yolo_detection_node:main',
            'object_hunt = erc_gazebo_sensors_py.object_hunt:main'
        ],
    },
)
