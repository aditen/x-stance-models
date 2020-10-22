call echo Starting setup. Works only on windows x64 machines!
call python --version
call python -m venv venv
call venv/Scripts/activate.bat
call pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
call pip install -r requirements.txt
call python setup.py
call python train_tiny_model.py
call echo Finished setup. Congrats!
pause
