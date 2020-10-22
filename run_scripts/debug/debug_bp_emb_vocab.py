import os

from models.debugging_utils import XStanceDebuggingUtils
from models.pipelines.embedding.bpemb_pipeline import BpembPipeline

if __name__ == "__main__":
    print("Starting")
    os.chdir("../..")
    pipeline = BpembPipeline(100000, 3)
    q = "Befürworten Sie Bestrebungen in den Kantonen zur Senkung der Sozialhilfeleistungen"
    a = "Die Kantone sollen sich um Missbräuche durch Leistungsbezüger kümmern."
    XStanceDebuggingUtils.print_tokens_based_on_embedding(pipeline, pipeline.tokenize(q, a))
    q = "Sareste d'accordo che in Svizzera fosse autorizzata l'eutanasia diretta e attiva da parte di un medico?"
    a = "La vita ci è data e non è nostra, anche se abbiamo a possibilità di frane quel che vogliamo. Per questo va ancor più rispettata, specialmente nei momenti difficili e che sembrano irriversibili."
    XStanceDebuggingUtils.print_tokens_based_on_embedding(pipeline, pipeline.tokenize(q, a))
