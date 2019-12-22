FROM pytorch/pytorch:latest

# TODO use requirments.txt

RUN python -m pip install matplotlib scipy scikit-image sklearn Pillow pandas

RUN git clone https://github.com/FamousDirector/pycwt.git && cd pycwt && python setup.py install
