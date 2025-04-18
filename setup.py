from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="car_racing_rl",
    version="0.1.0",
    author="SNCA-24",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author_email="nc.lonestar.tx@gmail.com",
    url="https://github.com/SNCA-24/rl-carracing-dqn-vs-ppo",
    license="MIT",
    description="Deep RL agents (DQN variants + PPO) for OpenAI Gym CarRacing-v2",
    packages=find_packages(exclude=["tests*"]),
    # install_requires=[
    #     # "gymnasium[box2d]==0.29.1",
    #     "gymnasium==0.29.1"             
    #     "tensorflow==2.15.0",
    #     "stable-baselines3[extra]==2.3.0",
    #     "opencv-python",
    #     "numpy",
    #     "pandas",
    #     "matplotlib",
    #     "pyyaml",
    # ],
    install_requires=[
        # Core dependencies (no GPU/CUDA wheels by default)
        "gymnasium==0.29.0",
        "tensorflow>=2.15,<3.0",
        "opencv-python",
        "numpy",
        "pandas",
        "matplotlib",
        "pyyaml",
    ],
    extras_require={
        # For DQN‑family + basic env
        # fall back to `pip install car_racing_rl[sb3]` to get Stable‑Baselines3
        "sb3": [
            "stable-baselines3[extra]>=2.1,<3.0",
            "shimmy>=1.3.0",
        ],
        # For video recording
        "video": [
            "moviepy",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "car-racing-train = scripts.train:main",
            "car-racing-evaluate = scripts.evaluate:main",
            "car-racing-record = scripts.record_video:main",
            "car-racing-plot = scripts.plot_metrics:main",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)

