reactionrnn <- function() {
  weights_path <- system.file("extdata", "reactionrnn_weights.hdf5", package="reactionrnn")
  vocab_path <- system.file("extdata", "reactionrnn_vocab.json", package="reactionrnn")

  maxlen <- 140
  reactions <- c('love', 'wow', 'haha', 'sad', 'angry')
  tokenizer <- text_tokenizer(filters='', char_level=T)
  tokenizer$word_index <- rjson::fromJSON(file=vocab_path)
  num_classes = length(tokenizer$word_index) + 1
  model = reactionrnn_build(weights_path, num_classes)
  model_enc = keras_model(inputs = model$input,
                          outputs = get_layer(model, 'rnn')$output)
  structure(list(maxlen=maxlen,
                 reactions=reactions,
                 tokenizer=tokenizer,
                 num_classes=num_classes,
                 model=model,
                 model_enc=model_enc
                 ), class="reactionrnn")
}


predict.reactionrnn <- function(obj, texts) {
  texts_enc <- texts %>% as.list() %>% encode_sequences(obj$tokenizer)
  predictions <- obj$model %>% predict(texts_enc, batch_size=1)
  if (nrow(predictions) == 1) {
    predictions <- predictions %>% c() %>% setNames(obj$reactions) %>% sort(T)
  }
  else {
    predictions <- predictions %>% as.data.frame() %>% setNames(obj$reactions)
  }

  return(predictions)
}

encode <- function(object, ...) UseMethod("encode")

encode.reactionrnn <- function(obj, texts) {
  texts_enc <- texts %>% as.list() %>% encode_sequences(obj$tokenizer)
  predictions <- obj$model_enc %>% predict(texts_enc) %>% as.data.frame()
  return(predictions)
}

predict_label <- function(object, ...) UseMethod("predict_label")

predict_label.reactionrnn <- function(obj, texts) {
  texts_enc <- texts %>% as.list() %>% encode_sequences(obj$tokenizer)
  predictions <- obj$model %>% predict(texts_enc, batch_size=1) %>% apply(1, which.max)
  return(obj$reactions[predictions])
}
