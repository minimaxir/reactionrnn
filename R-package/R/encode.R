encode_sequences <- function(texts, tokenizer, maxlen=140) {
  if (!is.vector(texts)) {texts <- c(texts)}
  texts_enc <- tokenizer %>% texts_to_sequences(texts)
  texts_enc <- texts_enc %>% pad_sequences(maxlen=maxlen)
  return(texts_enc)
}
