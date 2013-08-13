library(textcat)
library(stringr)

setwd("~/Documents/test_FP_models/Google_API")

clean.text <- function(text){
  # INPUT: Text to be "cleansed"
  # OUTPUT: Cleansed text
  # USAGE: clean.text(text) will return a string that has the punctuation removed
  #        lower case, and all other text cleaning operations done
  replace.links <- function(text){
    # extract urls from string, only works with t.co links, which all links in
    # twitter are nowadays
    return(str_replace_all(text,
                           ignore.case("http://[a-z0-9].[a-z]{2,3}/[a-z0-9]+"),
                           "urlextracted"))
  }
  remove.word <- function(string, starts.with.char){
    # INPUT:  string is a string to be edited,
    #         starts.with.char is a string or partial string to search and remove
    # OUTPUT: string with words removed
    # USAGE:  remove.word(string, "@") removes words starting with "@"
    #         remove.word(string, "RT") removes RT from string
    word.len <- nchar(starts.with.char)
    list.of.words <- strsplit(string, " ")[[1]]
    # remove ones that start with "starts.with.char"
    list.of.words <- list.of.words[!substring(list.of.words, 1,
                                              word.len)==starts.with.char]
    ret.string <- paste(list.of.words, collapse=" ")
    return(ret.string)
  }
  
  text.cleansed <- tolower(text)
  # remove the string "food poisoning" because every tweet has this in it...
  text.cleansed <- gsub("food poisoning", "", text.cleansed)
  text.cleansed <- replace.links(text.cleansed)
  text.cleansed <- remove.word(text.cleansed, "@")
  text.cleansed <- remove.word(text.cleansed, "rt")
  # replace non-letters with spaces
  text.cleansed <- gsub("[^[:alnum:]]", " ", text.cleansed)
  # remove leading and trailing spaces
  text.cleansed <- gsub("^\\s+|\\s+$", "", text.cleansed)
  # replace multiple spaces next to each other with single space
  text.cleansed <- gsub("\\s{2,}", " ", text.cleansed)
  return(text.cleansed)
}  

# read the manually classified tweets
df <- read.csv("/home/rachael/Documents/test_FP_models/fp_tweets.csv", stringsAsFactors=F)

# do the preprocessing to the data. this needs to be done before any prediction
# using the model that is created.
df$text.cleansed <- as.character(sapply(df$text, clean.text))
df <- subset(df, !is.na(manual_class) & text.cleansed != "")
df <- subset(df, !duplicated(text.cleansed))

# train the model using the textcat package
fp.model <- textcat_profile_db(df$text.cleansed, df$manual_class)

# test it out...
textcat("i have the food poisoning", fp.model)

#---------------------------------------------------------------------
#---------------------------------------------------------------------

# train the model with 70% of the manually classified tweets
sample2 <- sample(1:nrow(df), 0.70*nrow(df))

train2 <- df[sample2,]
pred2 <- df[-sample2,]

fp.model <- textcat_profile_db(train2$text.cleansed, train2$manual_class)

# use the remaining 30% of the manually classified tweets for prediction
pred_class2 <- sapply(pred2$text.cleansed, function(x)textcat(x, fp.model))

pred2$pred_class2 <- pred_class2

#-------------------------------------------------------------------------
# Google Prediction API
export <- subset(df, select=c("manual_class, text.cleansed"))

export$manual_class[export$manual_class == "1"] <- "T"
export$manual_class[export$manual_class == "0"] <- "F"

write.csv(export, file = 'exported_FP.csv', row.names = F, col.names=FALSE)

# train the Google prediction API with the same 70% as we did with the fp.model
# pred the same remaining 30% with the google prediction API and compare
# make sure there are no column or row names, and that the manual_class column is first
export_train <- export[sample2,]
export_pred <- export[-sample2,]
write.csv(export_train, file = 'exported_train_FP.csv', row.names = F, col.names=FALSE)
write.csv(export_pred, file = 'exported_pred_FP.csv', row.names = F, col.names=FALSE)

# results from GoogleAPI
GoogleAPI <- read.csv("/home/rachael/Documents/test_FP_models/Google_API/exported_pred_FP.csv", stringsAsFactors=F, header = F)

stats <- function(data, pred, man){
  # pred is the column name for the column of predicted classifications
  # man is the column name for the manual classified column
  data$Accuracy[data$man == "1" & data$pred == "1"] <- "a"
  data$Accuracy[data$man == "0" & data$pred == "0"] <- "b"
  data$Accuracy[data$man == "1" & data$pred == "0"] <- "c"
  data$Accuracy[data$man == "0" & data$pred == "1"] <- "d"
  # Type I error: food poisoning tweet that model prediction classifies as 0
  # Type II error: non-food poisoning tweet that model prediction classifies as 1
  # a and b are correct classifications
  # c is false negatives and d is false postivies
  accuracy <- table(data$Accuracy)
  error <- (accuracy[3] + accuracy[4]) / sum(accuracy)
  accuracy_rate <- (accuracy[1]+accuracy[2]) / sum(accuracy)
 
  sensitivity <- accuracy[1] / (accuracy[1] + accuracy[3])
  specificity <- accuracy[2] / (accuracy[2] + accuracy[4])

  data_stats <- c(accuracy_rate, error, sensitivity, specificity)
  names(data_stats) <- c("Accuracy Rate", "Error Rate", "Sensitivity","Specificity") 
  return(data_stats)
}

stats_fp.model <- stats(pred2, manual_class, pred_class_2)
# .7209 Accuracy, .7778 Sensitivity, .6667 Specificity
stats_Google <- stats(Google_API, V1, V2)
