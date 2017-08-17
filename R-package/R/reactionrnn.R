reactionrnn <- function() {
  maxlen <- 140
  reactions <- c('love', 'wow', 'haha', 'sad', 'angry')
  tokenizer <- text_tokenizer(filters='', char_level=T)
  tokenizer$word_index <- fromJSON(file="data/reactionrnn_vocab.json")
  num_classes = length(tokenizer$word_index) + 1
  model = reactionrnn_build("data/reactionrnn_weights.hdf5", num_classes)
  #model_enc = keras_model(inputs = model %>% get_layer('input'),
  #                        outputs = model %>% get_layer('rnn'))
  structure(list(maxlen=maxlen,
                 reactions=reactions,
                 tokenizer=tokenizer,
                 num_classes=num_classes,
                 model=model
                 #model_enc
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
