rm(list=ls())
dataset <- read.csv('/Users/molly1998/Desktop/python/Tweets.csv', header=T)
nrow(dataset)
head(dataset)
library(dplyr)
library(tidyverse) 
library(tidytext) 
library(topicmodels) 
library(tm) 
library(SnowballC) 
library(reshape2)
library(wordcloud)
data_focus=function(data, company,neg_pos){
  data_p1=data[data[,'airline']==company,]
  if(neg_pos=='negative'){
    data_p2=data_p1[data_p1[,'airline_sentiment']==neg_pos,]
  }
  else{
    data_p2=data_p1[data_p1[,'airline_sentiment']!=neg_pos,]
  }
  data_p2
}

preprocess=function(data,custom_stop_words){
  reviewsCorpus <- Corpus(VectorSource(data$text)) 
  reviewsDTM <- DocumentTermMatrix(reviewsCorpus)
  reviewsDTM_tidy <- tidy(reviewsDTM)
  reviewsDTM_tidy_cleaned <- reviewsDTM_tidy %>% anti_join(stop_words, by = c("term" = "word")) %>% 
    anti_join(custom_stop_words, by = c("term" = "word"))
  cleaned_documents <- reviewsDTM_tidy_cleaned %>%
    group_by(document) %>% 
    mutate(terms = toString(rep(term, count))) %>%
    select(document, terms) %>%
    unique()
  return(cleaned_documents)
}

top_terms_by_LDA <- function(input_text, # should be a columm from a dataframe
                             number_of_topics) # number of topics (4 by default)
{    
  # create a corpus (type of object expected by tm) and document term matrix
  Corpus <- Corpus(VectorSource(input_text)) # make a corpus object
  DTM <- DocumentTermMatrix(Corpus) # get the count of words/document
  
  # remove any empty rows in our document term matrix (if there are any 
  # we'll get an error when we try to run our LDA)
  unique_indexes <- unique(DTM$i) # get the index of each unique value
  DTM <- DTM[unique_indexes,] # get a subset of only those indexes
  
  # preform LDA & get the words/topic in a tidy text format
  lda <- LDA(DTM, k = number_of_topics, control = list(seed = 1234))
  topics <- tidy(lda, matrix = "beta")
  
  # get the top ten terms for each topic
  top_terms <- topics  %>% # take the topics data frame and..
    group_by(topic) %>% # treat each topic as a different group
    top_n(10, beta) %>% # get the top 10 most informative words
    ungroup() %>% # ungroup
    arrange(topic, -beta) # arrange words in descending informativeness
  
  cloud=wordcloud(words = top_terms$term, freq = top_terms$beta, min.freq = 1,
                  max.words=200, random.order=FALSE, rot.per=0.35, 
                  colors=brewer.pal(8, "Dark2"))
  
  # plot the top ten terms for each topic in order
  plot=top_terms %>% # take the top terms
    mutate(term = reorder(term, beta)) %>% # sort terms by beta value 
    ggplot(aes(term, beta, fill = factor(topic))) + # plot beta by theme
    geom_col(show.legend = FALSE) + # as a bar plot
    facet_wrap(~ topic, scales = "free") + # which each topic in a seperate plot
    labs(x = NULL, y = "Beta") + # no x label, change y label 
    coord_flip() # turn bars sideways
  list(cloud,plot)
}

try1=dataset[dataset[,'airline']=='Virgin America',]
custom_stop_words <- tibble(word = c("@virginamerica", "@united", "@southwestair","@jetblue","@usairways","@americanair","@american","flights","flight"))
try1_p=preprocess(try1,custom_stop_words)
cloud=top_terms_by_LDA(try1_p$terms,2)

cloud
plot

