import json

from flask import Flask, Response, request
from flask_cors import CORS

from models.pipelines.evaluation.bow_evaluation_pipeline import BoWEvaluationPipeline
from models.pipelines.evaluation.self_attention_evaluation_pipeline import AttentionEvaluationPipeline
from models.properties.attention_model_properties import AttentionModelProperties
from models.properties.bow_properties import BoWModelProperties

app = Flask(__name__)
CORS(app)
bow_evaluation: BoWEvaluationPipeline = BoWEvaluationPipeline(BoWModelProperties.tiny_own_embedding())
attention_evaluation: AttentionEvaluationPipeline = AttentionEvaluationPipeline(
    AttentionModelProperties.small_bertrand())


@app.route('/predict_bow', methods=['POST'])
def predict_stance_bow():
    request_body = json.loads(request.get_data().decode("utf-8"))
    return Response(json.dumps(bow_evaluation.evaluate(request_body['question'], request_body['comment'])),
                    mimetype="application/json")


@app.route('/predict_attention', methods=['POST'])
def predict_stance_attention():
    request_body = json.loads(request.get_data().decode("utf-8"))
    return Response(json.dumps(attention_evaluation.evaluate(request_body['question'], request_body['comment'])),
                    mimetype="application/json")


@app.route('/health', methods=['GET'])
def health():
    res = {"health": "ok"}
    return Response(json.dumps(res), mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
