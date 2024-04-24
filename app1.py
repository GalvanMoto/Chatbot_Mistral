import os
from flask import Flask, request, jsonify
from llama_cpp import Llama

# Set the CUDA device to be used (assuming CUDA is properly set up)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify the GPU device index

app = Flask(__name__)

# Initialize the Llama model to run entirely on GPU
llm = Llama(
    model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=32768,
    n_threads=0,
    n_gpu_layers=32
)

@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data['prompt']

    # Perform inference with the prompt
    output = llm(
        f"<s>[INST] {prompt} [/INST]",
        max_tokens=1000,
        stop=["</s>"],
        echo=True
    )

    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(port=8000)  # Run the Flask app on port 5000