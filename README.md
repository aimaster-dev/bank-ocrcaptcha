# Bank OCR CAPTCHA Solver API

This project is a deep learning-based OCR solution designed to automatically solve CAPTCHA images from various Vietnamese banks including MB Bank, BIDV, Vietcombank, and Eximbank. It uses TensorFlow/Keras with CTC decoding and a Flask-based API for real-time CAPTCHA recognition.

## Features

- Trained OCR models per bank (MB Bank, BIDV, Eximbank, Vietcombank)
- Flask REST API endpoints for CAPTCHA solving
- Supports base64-encoded PNG images
- Uses TensorFlow, Keras, and custom CTC loss layer
- Character-level prediction with bidirectional LSTM layers

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-repo/bank-captcha.git
   cd bank-captcha
   ````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:

   ```bash
   python app.py
   ```

## API Usage

POST to one of the following endpoints with JSON body:

```json
{
  "base64": "your_image_data_base64"
}
```

* `/api/captcha/mbbank`
* `/api/captcha/bidv`
* `/api/captcha/eximbank`
* `/api/captcha/vietcombank`

## Example Response

```json
{
  "status": "success",
  "captcha": "smcm"
}
```
Use the provided training script to preprocess data, define and train the model, and export it using JSON and weight files. Models can be reloaded using `LoadModel()`.

## License

MIT License
