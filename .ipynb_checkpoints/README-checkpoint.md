# piano-transcribe
Original code is provided by the team from ByteDance [[pdf]](https://arxiv.org/pdf/2010.01815.pdf)  


## Usage

1. Install dependencies:  

```
pip install -r requirements.txt
```

2. You may want to change the designated GPU for transcription in `inference.py`  

3. If you have multiple GPUs to run, you can either:  
    - Change the code lines 58-62 in `piano_transcription_inference/inference.py` to run parallel on multiple GPUs for faster transcription  
    - To run with the original package directly:   
    ```
    pip install piano_transcription_inference
    ```

4. To start transcribing with pre-trained model:  
```
python3 inference.py
```