rm(list=ls())
dataset <- read.csv('Tweets.csv', header=T)
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
library(data.table)
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
    reviewsCorpus <- Corpus(VectorSource(dataset$text)) 
    reviewsDTM <- DocumentTermMatrix(reviewsCorpus)
    reviewsDTM_tidy <- tidy(reviewsDTM)
    reviewsDTM_tidy_cleaned <- reviewsDTM_tidy %>% anti_join(stop_words, by = c("term" = "word")) %>% 
        anti_join(custom_stop_words, by = c("term" = "word"))
    cleaned_documents <- reviewsDTM_tidy_cleaned %>%
        group_by(document) %>% 
        mutate(terms = toString(rep(term, count))) %>%
        select(document, terms) %>%
        unique()
    processed_set=copy(dataset)
    m=length(cleaned_documents$terms)
    processed_set=processed_set[1:m,]
    processed_set$text=cleaned_documents$terms
    #processed_set$airline_sentiment=dataset$airline_sentiment#[:length(cleaned_documents$terms)]
    #processed_set$airline=dataset$airline#[:length(cleaned_documents$terms)]
    return(processed_set)
}

custom_stop_words <- tibble(word = c("@virginamerica", "@united", "@southwestair","@jetblue","@usairways","@americanair","@american","flights","flight"))
processed<-preprocess(dataset,custom_stop_words)

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

#try1=data_focus(dataset,'Virgin America','negative')
#custom_stop_words <- tibble(word = c("@virginamerica", "@united", "@southwestair","@jetblue","@usairways","@americanair","@american","flights","flight"))
#try1_p=preprocess(try1,custom_stop_words)
#cloud=top_terms_by_LDA(try1_p$terms,2)

#cloud
#plot

library(shiny)
library(leaflet)
library(Rmisc)

ui<-navbarPage('Data-driven Analysis for Airline',inverse = T,collapsible = T,
               tabPanel("Topic analysis ",
                        h1(span("Find why customers are satisfied/unsatisfied from comments.", style = "font-weight: 300"), 
                           style = "font-family: 'Source Sans Pro'; text-align: center;padding: 15px"),
                        sidebarLayout(position = "right",
                                      sidebarPanel(
                                          p("Please choose the airline:"),
                                          selectInput(inputId = "airline", label = strong("Airline"),choices = c("All", as.character(unique(dataset$airline)))),
                                          p("Please choose negtive-topics or positive-topics:"),
                                          selectInput(inputId = "neg_pos", "Reason for:", choices = c("All", as.character(unique(dataset$airline_sentiment))))
                                      ),
                                      mainPanel(
                                          tabsetPanel(
                                              tabPanel("Topics in comments",
                                                       
                                                       plotOutput(outputId = "charts",width = '100%', height = "300px"),
                                                       textOutput(outputId = "describ1")
                                                       #plotOutput(outputId = "clouds",width = '100%', height = "300px"),
                                                       #textOutput(outputId = "describ2")
                                              )
                                              
                                          )
                                      ))),
               tabPanel("Airline Reviews and Scoring",
                        h1(span("Most recent reviews and over-all score distribution", style = "font-weight: 300"), 
                           style = "font-family: 'Source Sans Pro'; text-align: center;padding: 15px"),
                        sidebarLayout(position = "right",
                                      sidebarPanel(
                                          p("Please choose the airline:"),
                                          selectInput(inputId = "airline", label = strong("Airline"),choices = c("All", as.character(unique(dataset$airline))))
                                      ),
                                      mainPanel(
                                          tabsetPanel(
                                              tabPanel("Most Recent Review",
                                                       
                                                       #plotOutput(outputId = "charts",width = '100%', height = "300px"),
                                                       tableOutput(outputId ='review_table')
                                                       #plotOutput(outputId = "clouds",width = '100%', height = "300px"),
                                                       #textOutput(outputId = "describ2")
                                              ),
                                              tabPanel("Score Distribution",
                                                       
                                                       plotOutput(outputId = "score_charts",width = '100%', height = "300px")
                                                       #tableOutput(outputId ='review_table')
                                                       #plotOutput(outputId = "clouds",width = '100%', height = "300px"),
                                                       #textOutput(outputId = "describ2")
                                              )
                                          )))
               )
               
)
server <- function(input, output) {
    output$score_charts <- renderPlot({
        if (input$airline=='All'){
            "Please select airline first."
        }
        else{
            try1=dataset[dataset[,'airline']==input$airline,]
            counts<-table(try1$airline_sentiment)
            #ylim_max=length(try1$airline)
            p1<-barplot(counts, main="Score Distribution",
                        xlab="Score",ylab = "Number")
            p1
        }
        
    })
    
    output$review_table <- renderTable({
        try1=dataset[dataset[,'airline']==input$airline,]
        review_table<-head(try1[order(try1$tweet_created,decreasing = TRUE),'text'],12)
        review_table
    })
    output$charts <- renderPlot({
        if (input$airline=='All'){
            "Please select airline first."
        }
        else{
            try1<-data_focus(processed,input$airline,input$neg_pos)
            Corpus <- Corpus(VectorSource(try1$text)) 
            DTM <- DocumentTermMatrix(Corpus) 
            unique_indexes <- unique(DTM$i) 
            DTM <- DTM[unique_indexes,] 
            
            lda <- LDA(DTM, k = 2, control = list(seed = 1234))
            topics <- tidy(lda, matrix = "beta")
            
            top_terms <- topics  %>% # take the topics data frame and..
                group_by(topic) %>% # treat each topic as a different group
                top_n(10, beta) %>% # get the top 10 most informative words
                ungroup() %>% # ungroup
                arrange(topic, -beta) # arrange words in descending informativeness
            
            plot=top_terms %>% # take the top terms
                mutate(term = reorder(term, beta)) %>% # sort terms by beta value 
                ggplot(aes(term, beta, fill = factor(topic))) + # plot beta by theme
                geom_col(show.legend = FALSE) + # as a bar plot
                facet_wrap(~ topic, scales = "free") + # which each topic in a seperate plot
                labs(x = NULL, y = "Topics") + # no x label, change y label 
                coord_flip() # turn bars sideways
            plot
        }
        
    })
    output$describ1 <- renderText({
        trend_text <- input$neg_pos
        airline_text<-input$airline
        paste("The key words in",trend_text,"comments about",airline_text)
    })
    
    
    
}
shinyApp(ui = ui, server = server)
