from setuptools import setup

package_name = 'imu_processor_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='你的名字',
    maintainer_email='你的邮箱@example.com',
    description='处理IMU数据并计算角加速度的包',
    license='License declaration',  # 在这里填写许可协议，例如 'Proprietary' 或 'Apache License 2.0'
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_processor = imu_processor_pkg.imu_processor:main',
        ],
    },
)
