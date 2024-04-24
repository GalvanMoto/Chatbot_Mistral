from flask import Flask, request, jsonify
#from llama_cpp import Llama
from langchain.llms import LlamaCpp  
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from torch import cuda

app = Flask(__name__)
model_path = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
n_gpu_layers = 48  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# Initialize the Llama model
# #
# llm.to(cuda.current_device())
@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data['prompt']

    # Perform inference with the prompt
    output = llm(
        f"<s>[INST] {prompt} [/INST]",
        max_tokens=1000,
        stop=["</s>"]
    )

    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True,port=5000)  # Run the Flask app on port 5000