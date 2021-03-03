rm(list=ls())
dataset <- read.csv('Tweets.csv', header=T)[ ,c('airline','airline_sentiment','tweet_created','text')]
library(dplyr)
library(tidyverse) 
library(tidytext) 
library(topicmodels) 
library(tm) 
library(SnowballC) 
library(reshape2)
library(wordcloud)

library(shiny)
library(leaflet)
library(Rmisc)

ui<-navbarPage('Data-driven Analysis for Airline',inverse = T,collapsible = T,
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
        try1=dataset[dataset[,'airline']==input$airline,]
        counts<-table(try1$airline_sentiment)
        #ylim_max=length(try1$airline)
        p1<-barplot(counts, main="Score Distribution",
                    xlab="Score",ylab = "Number")
        p1
    })
    
    output$review_table <- renderTable({
        try1=dataset[dataset[,'airline']==input$airline,]
        review_table<-head(try1[order(try1$tweet_created,decreasing = TRUE),'text'],12)
        review_table
    })
    
    
}

shinyApp(ui = ui, server = server)

