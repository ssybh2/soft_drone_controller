from setuptools import setup
import os
from glob import glob

package_name = 'soft_drone_controller'

packages = [
    package_name,  # 原有的soft_drone_controller模块
    f'{package_name}.config'  # 新增：config子模块
]

setup(
    name=package_name,
    version='0.0.0',
    packages=packages,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 安装配置文件
     
        # 安装launch文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'scipy', 'numpy'],
    zip_safe=True,
    maintainer='Boyang Hu',
    maintainer_email='ssybh2@nottingham.edu.cn',
    description='SoftDrone ROS2 Controller with EtherCAT',
    license='Apache-2.0',
    tests_require=['pytest'],
    # 节点入口（终端可直接启动）
    entry_points={
        'console_scripts': [
            'drone_controller = soft_drone_controller.drone_controller:main',
            'position_control = soft_drone_controller.position_control:main',
            'send_target = soft_drone_controller.send_target:main', 
            'position_cmd = soft_drone_controller.position_cmd:main',
            'pos_path_to_nav_path = soft_drone_controller.pos_path_to_nav_path:main',
            
        ],
    },
)

