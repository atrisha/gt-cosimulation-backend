library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(tidyverse)
library(ggpubr)
library(data.table)


rm(list=ls())

#Merge before intersection

mbi_dataframe <- read.csv(file = 'D:\\repeated_games_data\\intersection_dataset\\synthetic\\merge_before_intersection\\results\\all_results.csv')

nrow(mbi_dataframe[mbi_dataframe$model_type == 'robust_resp' & mbi_dataframe$agent1_type>=2 & mbi_dataframe$agent2_type>=2, ])
nrow(mbi_dataframe[mbi_dataframe$model_type == 'auto_resp' & mbi_dataframe$agent1_type>=2 & mbi_dataframe$agent2_type>=2, ])/table(mbi_dataframe$model_type)['auto_resp']

td <- mbi_dataframe
td <- mbi_dataframe[mbi_dataframe$agent1_model_type == mbi_dataframe$agent2_model_type, ]
mbi_dataframe <- td
td$type_comb <- paste(td$agent1_type,td$agent2_type)
td2 <- ddply(td, .(agent1_model_type,type_comb), function(td) c(count=nrow(td)))
ggplot(td2, aes(x=type_comb,y=count, group=agent1_model_type)) +geom_line(aes(color=agent1_model_type)) +geom_point(aes(color=agent1_model_type)) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggplot(td, aes(x=type_comb)) +geom_bar(aes(fill = agent1_model_type)) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggplot(mbi_dataframe,aes(x=agent2_type,group=agent1_model_type,fill=agent1_model_type))+geom_histogram(position="dodge",binwidth=0.25)+theme_bw()
f1 <- ggplot(td, aes(x=type_comb)) +geom_bar(aes(fill = agent1_model_type)) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
# Plot corresponding heatmap
td2 <- td[2:4] %>% rownames_to_column()
td2$rowname <- paste(td2$agent1_type ,td2$agent2_type)
td2 <- gather(td2, ag_id, value, agent1_type:agent2_type, factor_key=TRUE)
f2 <- ggplot(td2, aes(x = rowname, y = ag_id, fill = value)) + geom_tile() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), axis.text.y=element_blank())
ggarrange(f1, f2,ncol = 1, nrow = 2)


td <- transform(td, value = ave(agent1_model_type , agent1_type, agent2_type , FUN = length))
td$value <- as.numeric(td$value)

type_var_df <- data.frame(model=factor(),mean=double(),sd=double())
names(type_var_df)<-c("model","mean","sd")
td2$agent1_model_type <- as.factor(td2$agent1_model_type)
for (m in levels(td2$agent1_model_type)) {
  x <- data.frame(m,mean(td2[td2$agent1_model_type==m,]$count),sd(td2[td2$agent1_model_type==m,]$count))
  names(x)<-c("model","mean","sd")
  type_var_df <- rbind(type_var_df,x)
}
ggplot(type_var_df, aes(model, mean)) + geom_point() + geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd))
ggplot(type_var_df, aes(x=mean, y=sd, color=model)) + geom_point() +     geom_text(label=type_var_df$model) + labs(x='mean success count across agent types',y='SD of success count across agent types')


td <- mbi_dataframe[mbi_dataframe$dist_gap  > 2.5, ]
td$strat_type = "high_distance_gap"
t <- mbi_dataframe[mbi_dataframe$dist_gap  <= 2.5, ]
t$strat_type = "low_distance_gap"
td <- rbind(td,t)






# parking pullout
pp_dataframe <- read.csv(file = 'D:\\repeated_games_data\\intersection_dataset\\synthetic\\parking_pullout\\results\\all_results.csv')

td <- pp_dataframe
td <- pp_dataframe[pp_dataframe$agent1_model_type == pp_dataframe$agent2_model_type, ]
td$type_comb <- paste(td$agent1_type,td$agent2_type)
td2 <- ddply(td, .(agent1_model_type ,type_comb), function(td) c(count=nrow(td)))
ggplot(td2, aes(x=type_comb,y=count, group=agent1_model_type )) +geom_line(aes(color=agent1_model_type )) +geom_point(aes(color=agent1_model_type )) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
ggplot(td, aes(x=type_comb)) +geom_bar(aes(fill = agent1_model_type )) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

ggplot(mbi_dataframe,aes(x=agent2_type,group=agent1_model_type ,fill=agent1_model_type ))+geom_histogram(position="dodge",binwidth=0.25)+theme_bw()
f1 <- ggplot(td, aes(x=type_comb)) +geom_bar(aes(fill = agent1_model_type )) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
# Plot corresponding heatmap
td2 <- td[2:4] %>% rownames_to_column()
td2$rowname <- paste(td2$agent1_type ,td2$agent2_type)
td2 <- gather(td2, ag_id, value, agent1_type:agent2_type, factor_key=TRUE)
f2 <- ggplot(td2, aes(x = rowname, y = ag_id, fill = value)) + geom_tile() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), axis.text.y=element_blank())
ggarrange(f1, f2,ncol = 1, nrow = 2)
t <- as.data.frame(td[(td$agent1_model_type == 'robust_resp' & td$agent2_model_type == 'robust_resp') | (td$agent1_model_type == 'ql1_resp' & td$agent2_model_type == 'ql1_resp') | td$agent1_model_type == 'mspe' | td$agent1_model_type == 'uspe', ])
ggplot(t, aes(x=dist_gap , fill=agent1_model_type)) + geom_density(aes(y=..density..), alpha=0.5)


# Intersection clearance

ic_dataframe <- read.csv(file = 'D:\\repeated_games_data\\intersection_dataset\\synthetic\\intersection_clearance\\results\\all_results.csv')

td <- ic_dataframe
td <- ic_dataframe[ic_dataframe$agent1_model_type == ic_dataframe$agent2_model_type, ]
ic_dataframe <- td
td$model_type <- td$agent1_model_type

td <- ic_dataframe[ic_dataframe$st1_trajl > 20 & ic_dataframe$st2_trajl < 50, ]
ggplot(td, aes(x=agent1_model_type)) +geom_bar()

td <- ic_dataframe[ic_dataframe$st1_trajl > 45 & ic_dataframe$st2_trajl > 70, ]
ggplot(td, aes(x=agent1_model_type)) +geom_bar()

td <- transform(td, value = ave(agent1_model_type , lt_type , st1_type ,st2_type  , FUN = length))
td$value <- as.numeric(td$value)

type_var_df <- data.frame(model=factor(),mean=double(),sd=double())
names(type_var_df)<-c("model","mean","sd")
td2$agent1_model_type <- td2$model_type
td2$agent1_model_type <- as.factor(td2$agent1_model_type)
for (m in levels(td2$agent1_model_type)) {
  x <- data.frame(m,mean(td2[td2$agent1_model_type==m,]$count),sd(td2[td2$agent1_model_type==m,]$count))
  names(x)<-c("model","mean","sd")
  type_var_df <- rbind(type_var_df,x)
}
ggplot(type_var_df, aes(model, mean)) + geom_point() + geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd))

td$type_comb <- paste(td$lt_type,td$st1_type,td$st2_type)
td2 <- ddply(td, .(model_type,type_comb), function(td) c(count=nrow(td)))
# Plot the barplot by model type and agent type

f1 <- ggplot(td, aes(x=type_comb)) +geom_bar(aes(fill = model_type)) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
# Plot corresponding heatmap
td2 <- td[3:5] %>% rownames_to_column()
td2$rowname <- paste(td2$lt_type,td2$st1_type,td2$st2_type)
td2 <- gather(td2, ag_id, value, lt_type:st2_type, factor_key=TRUE)
f2 <- ggplot(td2, aes(x = rowname, y = ag_id, fill = value)) + geom_tile() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), axis.text.y=element_blank())
ggarrange(f1, f2,ncol = 1, nrow = 2)

td <- ic_dataframe[ic_dataframe$st1_trajl > 20 & ic_dataframe$st2_trajl < 50, ]
td$strat_type = "crosses,waits"
t <- ic_dataframe[ic_dataframe$st1_trajl > 20 & ic_dataframe$st2_trajl > 45, ]
t$strat_type = "crosses,crosses"
td <- rbind(td,t)


t <- as.data.frame.matrix(table(td$agent1_model_type ,td$strat_type))
t$model_type <- rownames(t)
t2 <- gather(t, strat_type, count, `crosses,crosses`, `crosses,waits`, factor_key=TRUE)
ggplot(data=t2, aes(x=strat_type, y=count, fill=model_type)) + geom_bar(stat="identity")


td <- ic_dataframe
td$dist_gap12 <- as.numeric(td$dist_gap12)
td$dist_gap13 <- as.numeric(td$dist_gap13)
td <- transform(td, min_distgap = pmin(dist_gap12, dist_gap13))
td$model_type <- td$agent1_model_type
t <- as.data.frame(td[td$model_type == 'robust_resp' | td$model_type == 'ql1_resp' | td$model_type == 'mspe' | td$model_type == 'uspe' | td$model_type == 'auto_resp', ])
ggplot(t, aes(x=min_distgap, fill=agent1_model_type)) + geom_density(aes(y=..density..), alpha=0.5)


