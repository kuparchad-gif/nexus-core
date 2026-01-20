# Run: python -m src.tests.qwen_smoke
import os
import subprocess


from src.service.cogniKubes.berts.bert_layer_fixed import BertLayerStub

def main():
    stub = BertLayerStub("src/service/cognikubes/berts/qwen_config.json")
    print("process_input ->", stub.process_input("Optimize node scheduling for low GPU memory footprint"))
    print("classify ->", stub.classify("Route inbound task to cheapest resource", ["local", "cloud", "hybrid"]))

if __name__ == "__main__":
    main()
