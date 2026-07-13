from setuptools import setup, find_packages

setup(
    name='imgtrans',
    version='0.1',
    packages=find_packages(),
    py_modules=['imgtrans'],  # imgtrans.py シム。easing/hdr_io は imgtrans_lib パッケージ内へ移設済み
    install_requires=[
        'opencv-python',
        'numpy',
        'scipy',   # interpolate (splrep/splev/interp1d) を _dm_data_ops で使用
        'psutil',
        'matplotlib',
        'av',      # PyAV: HDR/10bit+ decode & YUV-native encode path
        'numba',   # JIT acceleration for YUV-native slit-scan (optional but recommended)
    ],
    # NOTE: FFmpeg (ffmpeg & ffprobe) must also be installed on the system PATH.
    # その他のメタデータ（必要に応じて）
    author='Ryu Furusawa',
    author_email='ryufurusawa@gmail.com',
    description='A programming tool for manipulating time and space of video data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ryufurusawa/imgtrans',
    classifiers=[
        'Development Status :: 3 - Alpha',  # または '4 - Beta'、'5 - Production/Stable' など
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
    ],
)