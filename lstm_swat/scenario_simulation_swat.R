
library(tidyverse)
library(hydroGOF)
library(data.table)

wd <- "C:/Users/nguyenta/Documents/Manuscript/lstm_swat"

#------------------------------------------------------------------------------#
#              Prepared input data for the LSTM                                #
#------------------------------------------------------------------------------#
setwd(wd)
meterological <- fread("data/SWAT/observed_meterological.csv")

# Add basin average eta to meterological data
output_sub <- fread("data/SWAT/workingFolder_best/TxtInOut_1/output.sub",
                    header = FALSE, skip = 9)

# Get basin average
output_sub <- output_sub %>%
  reframe(time = paste0(V6, "-", sprintf("%02d",V4), "-", 
                        sprintf("%02d", V5)),
          eta_area = V11 * V7) %>%
  group_by(time) %>%
  summarise(eta_mm = sum(eta_area)/area) 

meterological$eta_mm <- round(output_sub$eta_mm, 3)
meterological$q_mm <- round(fread("data/SWAT/workingFolder_best/TxtInOut_1/watout.dat",
              header = FALSE, skip = 6)$V4 * 86.4/2176.453, 3)

fwrite(meterological, file = "data/LSTM/time_series.csv")
#------------------------------------------------------------------------------#
#                            Random permutation                                #
#------------------------------------------------------------------------------#
setwd(wd)
pcp <- readLines("data/SWAT/workingFolder_best/TxtInOut_random_permutation/pcp1_orig.pcp")
temp <- readLines("data/SWAT/workingFolder_best/TxtInOut_random_permutation/Tmp1_orig.Tmp")
q_orig <- read.csv("data/SWAT/workingFolder_best/TxtInOut_1/watout.dat", 
                   skip=6, header=FALSE, sep ="")$V4[-c(1:12784)]
eta_orig <- fread("data/SWAT/workingFolder_best/TxtInOut_1/output.sub", 
                  header = FALSE, skip = 9)
eta_orig <- eta_orig %>% reframe(time = paste0(V6, "-", sprintf("%02d",V4), "-",
                        sprintf("%02d", V5)), eta_area = V11 * V7) %>%
  group_by(time) %>% summarise(eta_mm = sum(eta_area)/area) 
eta_orig <- eta_orig$eta_mm[12785:16071]

# Pcp
nse_pcp_q <- c()
nse_pcp_eta <- c()

setwd(file.path(wd, "data/SWAT/workingFolder_best/TxtInOut_random_permutation"))
for (iter in 1:100){
  idx <- c(c(1:4), sample(c(5:length(pcp)), replace=FALSE))
  pcp_new <- pcp[idx]
  substr(pcp_new, 1,7) <- substr(pcp, 1,7)
  writeLines(pcp_new,"pcp1.pcp")
  
  system("swat.exe")
  
  q <- read.csv("watout.dat", skip=6, header=FALSE, sep ="")$V4
  
  # Read eta
  eta <- fread("output.sub", header = FALSE, skip = 9)
  eta <- eta %>% 
    reframe(time = paste0(V6, "-", sprintf("%02d",V4), "-",
                          sprintf("%02d", V5)), 
            eta_area = V11 * V7) %>%
    group_by(time) %>%
    summarise(eta_mm = sum(eta_area)/area) 
  
  
  nse_pcp_q <- c(nse_pcp_q, NSE(q, q_orig))
  nse_pcp_eta <- c(nse_pcp_eta, NSE(eta$eta_mm, eta_orig))
}


# Tempearature min
nse_tmin_q <- c()
nse_tmin_eta <- c()

for (iter in 1:100){
  idx <- c(c(1:4), sample(c(5:length(pcp)), replace=FALSE))
  temp_new <- temp[idx]
  # date
  substr(temp_new, 1,7) <- substr(temp, 1,7)
  
  # tmax station 1-4
  substr(temp_new, 8,12) <- substr(temp, 8,12)
  substr(temp_new, 18,22) <- substr(temp, 18,22)
  substr(temp_new, 28,32) <- substr(temp, 28,32)
  substr(temp_new, 38,42) <- substr(temp, 38,42)

  
  writeLines(temp_new,"Tmp1.Tmp")
  
  system("swat.exe")
  
  q <- read.csv("watout.dat", skip=6, header=FALSE, sep ="")$V4

  # Read eta
  eta <- fread("output.sub", header = FALSE, skip = 9)
  eta <- eta %>% 
    reframe(time = paste0(V6, "-", sprintf("%02d",V4), "-",
                          sprintf("%02d", V5)), 
            eta_area = V11 * V7) %>%
    group_by(time) %>%
    summarise(eta_mm = sum(eta_area)/area) 
  
  
  nse_tmin_q <- c(nse_tmin_q, NSE(q, q_orig))
  nse_tmin_eta <- c(nse_tmin_eta, NSE(eta$eta_mm, eta_orig))
  
  
}

# Tempearature max
nse_tmax_q <- c()
nse_tmax_eta <- c()

for (iter in 1:100){
  idx <- c(c(1:4), sample(c(5:length(pcp)), replace=FALSE))
  temp_new <- temp[idx]
  # date
  substr(temp_new, 1,7) <- substr(temp, 1,7)
  
  # tmax station 1-4
  substr(temp_new, 13,17) <- substr(temp, 13,17)
  substr(temp_new, 23,27) <- substr(temp, 23,27)
  substr(temp_new, 33,37) <- substr(temp, 33,37)
  substr(temp_new, 43,47) <- substr(temp, 43,47)
  
  
  writeLines(temp_new,"Tmp1.Tmp")
  
  system("swat.exe")
  
  q <- read.csv("watout.dat", skip=6, header=FALSE, sep ="")$V4

  # Read eta
  eta <- fread("output.sub", header = FALSE, skip = 9)
  eta <- eta %>% 
    reframe(time = paste0(V6, "-", sprintf("%02d",V4), "-",
                          sprintf("%02d", V5)), 
            eta_area = V11 * V7) %>%
    group_by(time) %>%
    summarise(eta_mm = sum(eta_area)/area) 
  
  
  nse_tmax_q <- c(nse_tmax_q, NSE(q, q_orig))
  nse_tmax_eta <- c(nse_tmax_eta, NSE(eta$eta_mm, eta_orig))
  
}



nse_combine <- tibble(nse_pcp_q = nse_pcp_q,
                      nse_pcp_eta = nse_pcp_eta,
                      nse_tmin_q = nse_tmin_q,
                      nse_tmin_eta = nse_tmin_eta,
                      nse_tmax_q = nse_tmax_q,
                      nse_tmax_eta = nse_tmax_eta)

write.csv(nse_combine, file = file.path(wd, "data/SWAT", "workingFolder_best", 
                                        "random_permutation_nse.csv"),
         quote = FALSE, row.names = FALSE)

#------------------------------------------------------------------------------#
#                            Lag time                                          #
#------------------------------------------------------------------------------#
setwd(wd)
pcp <- readLines("SWAT/workingFolder_best/TxtInOut_random_permutation/pcp1_orig.pcp")
q_orig <- read.csv("SWAT/workingFolder_best/TxtInOut_1/watout.dat", 
                   skip=6, header=FALSE, sep ="")$V4[-c(1:12784)]


setwd(file.path(wd, "SWAT/workingFolder_best/TxtInOut_lagtime"))
result <- matrix(rep(NA, 365*366), ncol = 366)
counter <- 1
for (i in c(735:1100)){
  print(counter)
  
  pcp_new <- pcp
  substr(pcp_new[i], 8, 27) <- "171.3171.3171.3171.3"
  writeLines(pcp_new,"pcp1.pcp")
  
  system("swat.exe")
  
  q <- read.csv("watout.dat", skip=6, header=FALSE, sep ="")$V4
  
  qdiff <- q - q_orig[1:length(q)]
  result[,counter] <- qdiff[(i-369):(i + 365-370)]
  
  counter <- counter + 1
}

write.csv(result, file = file.path(wd, "SWAT", "workingFolder_best", 
                              "runoff_timelag.csv"),
          quote = FALSE, row.names = FALSE)


#------------------------------------------------------------------------------#
#                            Zero precitpitaion                                #
#------------------------------------------------------------------------------#
setwd(wd)
pcp <- readLines("data/SWAT/workingFolder_best/TxtInOut_zero_pre/pcp1_orig.pcp")
setwd(file.path(wd, "data/SWAT/workingFolder_best/TxtInOut_zero_pre"))

substr(pcp_new[5:3656], 8, 27) <- "000.0000.0000.0000.0"
writeLines(pcp_new,"pcp1.pcp")
system("swat.exe")
q <- read.csv("watout.dat", skip=6, header=FALSE, sep ="")$V4
write.csv(q, file = file.path(wd, "data/SWAT", "workingFolder_best", 
                                   "runoff_zero_pre.csv"),
          quote = FALSE, row.names = FALSE)