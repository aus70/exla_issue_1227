defmodule ExlaIssue1227 do
  @model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

  def run(text, model_name \\ @model_name) do
    {:ok, model_info} = Bumblebee.load_model({:hf, model_name})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model_name})

    inputs = Bumblebee.apply_tokenizer(tokenizer, [text])
    Axon.predict(model_info.model, model_info.params, inputs, compiler: EXLA)
  end
end
