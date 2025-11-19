# project script

# libraries
library(ggplot2)
library(urca)
library(tseries)

data <- read.csv("data/UMCSENT.csv")

# 1. DATA PREPARATION ---------------------------------------------------------------------

str(data) # observation_date chr; UMCSENT num
data$observation_date <- as.Date(data$observation_date, format = "%Y-%m-%d") # observation_date conversion to Date
str(data)

sum(is.na(data$UMCSENT)) # check missing values
mean(is.na(data$UMCSENT)) # percentage of missing values

nrow(data)
range(data$observation_date)

ggplot(data, aes(x = observation_date, y = UMCSENT)) +
  geom_line() +
  labs(title = 'Consumer Sentiment Index')


# data from 1978
data_filtered <- subset(data, observation_date >= "1978-01-01")

nrow(data_filtered)
range(data_filtered$observation_date)
ggplot(data_filtered, aes(x = observation_date, y = UMCSENT)) +
  geom_line() +
  labs(title = 'Consumer Sentiment Index')

# statistics
summary(data_filtered$UMCSENT)

# check for outliers
hist(data_filtered$UMCSENT, breaks = 20)
boxplot(data_filtered$UMCSENT)


# STATIONARITY TEST ------------------------------------------------------------------
# ADF -> stationary
summary(ur.df(data_filtered$UMCSENT, type = "drift"))

# Conclusione su $\tau_2$:Confronto: 
# Il valore della statistica test ($-3.3225$) è più negativo del valore critico al 
# livello di significatività del 5% ($-2.86$).
# Decisione: Poiché $-3.3225 < -2.86$, si rigetta l'Ipotesi Nulla ($H_0$) al livello del 5%.
# Interpretazione: Si conclude che la serie storica UMCSENT è stazionaria (trend-stazionaria) al livello del 5%.
# Non è necessario differenziare la serie per renderla stazionaria se si include un termine di drift.

# KPSS -> non-stationary
kpss.test(data_filtered$UMCSENT)

# Il valore della statistica KPSS ($0.75044$) è maggiore sia del valore critico al 5% sia del valore critico all'1% ($\approx 0.739$).
# Si rigetta $H_0$ anche al livello dell'1%.


# AUTOCORRELATION --------------------------------------------------------------------
acf(data_filtered$UMCSENT)
pacf(data_filtered$UMCSENT)

# SEASONALITY OR TREND ----------------------------------------------------------------
data_ts <- ts(data_filtered$UMCSENT, start = c(1978, 01), frequency = 12)
plot(stl(data_ts, s.window="periodic"))


# FIRST DIFFERENCE -----------------------------------------------------------------------
diff_1 <- diff(data_filtered$UMCSENT)
acf(diff_1)
pacf(diff_1)

# correlation tests
bartlett.test(data_filtered$UMCSENT ~ as.factor(cut(seq_along(data_filtered$UMCSENT), 10))) # suggests log-transformation

Box.test(data_filtered$UMCSENT, lag = 10, type = "Box-Pierce") # high autocorrelation
Box.test(diff_1, lag = 10, type = "Box-Pierce")

Box.test(data_filtered$UMCSENT, lag = 10, type = "Ljung-Box") # high autocorrelation



#data filtered
data_filtered$log_UMCSENT <- log(data_filtered$UMCSENT)

diff_log <- diff(data_filtered$log_UMCSENT)

alligned_diff_log <- c(NA, diff_log)

data_filtered$diff_log_UMCSENT <- alligned_diff_log

head(data_filtered)

#acf and pacf of log data
acf(na.omit(data_filtered$diff_log_UMCSENT))
pacf(na.omit(data_filtered$diff_log_UMCSENT))


# data split
nrows <- nrow(data_filtered)
index_division <- floor(0.80 * nrows)

train <- subset(data_filtered, 1:nrows <= index_division)
test <- subset(data_filtered, 1:nrows > index_division)

cat("Training:", nrow(train), "rows\n")
cat("Test:", nrow(test), "rows\n")


# AIC e BIC -------------------------------------------------------------------
library(forecast)

# --- 1. CONFIGURAZIONE E INIZIALIZZAZIONE ---

# La serie di training da usare per la stima (sostituisce 'y')
y_series <- train$log_UMCSENT 
# Se 'y_series' non è un oggetto 'ts', è bene convertirlo per Arima() se non lo è già.
# Ad esempio:
# y_series <- ts(y_series, frequency = 12) 

# Inizializzazione delle matrici AIC e BIC (12 righe x 3 colonne)
aic <- matrix(NaN, nrow = 12, ncol = 3)
bic <- matrix(NaN, nrow = 12, ncol = 3)


# --- 2. CICLO PER AR(p) E MA(q) (p/q da 1 a 12) ---
for (ii in 1:12) {
  
  # A. Stima ARIMA(p, 1, 0)
  tryCatch({
    # USA: y_series (che è train$log_UMCSENT)
    mhat_ar <- Arima(y_series, order = c(ii, 1, 0), method = "ML", include.mean = FALSE)
    aic[ii, 1] <- mhat_ar$aic
    bic[ii, 1] <- mhat_ar$bic
  }, error = function(e) {
    # Non è necessario stampare l'errore a meno che non si voglia il debug
    # cat(paste("Errore nella stima AR(", ii, "): ", conditionMessage(e), "\n"))
  })
  
  # B. Stima ARIMA(0, 1, q)
  tryCatch({
    # USA: y_series
    mhat_ma <- Arima(y_series, order = c(0, 1, ii), method = "ML", include.mean = FALSE)
    aic[ii, 2] <- mhat_ma$aic
    bic[ii, 2] <- mhat_ma$bic
  }, error = function(e) {
    # cat(paste("Errore nella stima MA(", ii, "): ", conditionMessage(e), "\n"))
  })
}

# --- 3. CICLO PER ARMA(p, 1, q) MISTI (p=1,2 e q=1,2) ---
ii <- 1
for (pp in 1:2) {
  for (qq in 1:2) {
    # Stima ARIMA(p, 1, q)
    tryCatch({
      # USA: y_series
      mhat_arma <- Arima(y_series, order = c(pp, 1, qq), method = "ML", include.mean = FALSE)
      aic[ii, 3] <- mhat_arma$aic
      bic[ii, 3] <- mhat_arma$bic
      ii <- ii + 1
    }, error = function(e) {
      # Gestisce l'errore e passa al prossimo indice
      # cat(paste("Errore nella stima ARMA(", pp, ", 1, ", qq, "): ", conditionMessage(e), "\n"))
      ii <- ii + 1
    })
  }
}

# --- 4. PLOT DEI RISULTATI FILTRATI (ARIMA(p, 1, 0)) ---

# Controlla quali modelli AR(p) hanno prodotto risultati validi (non NaN)
valid_ar_models <- !is.nan(aic[, 1])

# Vettori per l'asse X (p=1 a p=12)
p_values <- 1:12

# Crea un DataFrame per il plotting filtrando solo i valori validi
df_plot_filtered <- data.frame(
  p = p_values[valid_ar_models],
  AIC = aic[valid_ar_models, 1],
  BIC = bic[valid_ar_models, 1]
)

library(ggplot2)

p <- ggplot(df_plot_filtered, aes(x = p)) +
  geom_line(aes(y = AIC, colour = "AIC")) +
  geom_point(aes(y = AIC, colour = "AIC"), shape = 1, size = 3) +
  geom_line(aes(y = BIC, colour = "BIC")) +
  geom_point(aes(y = BIC, colour = "BIC"), shape = 2, size = 3) +
  
  labs(
    title = "Criteri di Informazione per Modelli ARIMA(p, 1, 0)",
    x = "Ordine AR (p)",
    y = "Valore Criterio"
  ) +
  scale_colour_manual(name = "Criterio", values = c("AIC" = "red", "BIC" = "blue")) +
  # Assicura che l'asse X mostri tutti i valori p
  scale_x_continuous(breaks = p_values) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom"
  )

print(p)

library(ggplot2)

# --- 1. PREPARAZIONE DEI DATI FILTRATI (Colonna 2 per MA(q)) ---

# Controlla quali modelli MA(q) hanno prodotto risultati validi (non NaN)
valid_ma_models <- !is.nan(aic[, 2])

# Vettori per l'asse X (q=1 a q=12)
q_values <- 1:12

# Crea un DataFrame per il plotting filtrando solo i valori validi
df_plot_ma_filtered <- data.frame(
  q = q_values[valid_ma_models],
  AIC = aic[valid_ma_models, 2],
  BIC = bic[valid_ma_models, 2]
)

# --- 2. GENERAZIONE DEL GRAFICO ---

p_ma <- ggplot(df_plot_ma_filtered, aes(x = q)) +
  # Linea e punti per AIC
  geom_line(aes(y = AIC, colour = "AIC")) +
  geom_point(aes(y = AIC, colour = "AIC"), shape = 1, size = 3) +
  # Linea e punti per BIC
  geom_line(aes(y = BIC, colour = "BIC")) +
  geom_point(aes(y = BIC, colour = "BIC"), shape = 2, size = 3) +
  
  labs(
    title = "Criteri di Informazione per Modelli ARIMA(0, 1, q)",
    x = "Ordine MA (q)",
    y = "Valore Criterio"
  ) +
  # Imposta i colori e la legenda
  scale_colour_manual(name = "Criterio", values = c("AIC" = "red", "BIC" = "blue")) +
  # Assicura che l'asse X mostri tutti i valori q
  scale_x_continuous(breaks = q_values) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom"
  )

print(p_ma)


# BENCHMARK MODEL --------------------------------------------------------------------

model_rw <- arima(
  data_filtered$log_UMCSENT,
  order = c(0, 1, 0),
  method = "ML"
)

summary(model_rw)

full_series <- data_filtered$log_UMCSENT 
h_steps <- length(test$log_UMCSENT)

rolling_predictions <- numeric(h_steps)

n_train_initial <- length(train$log_UMCSENT)

for (i in 1:h_steps) {
  current_train_window <- full_series[1:(n_train_initial + i - 1)]
  current_model <- arima(
    current_train_window,
    order = c(0, 1, 0),
    method = "ML"
  )
  
  forecast_one_step <- forecast(current_model, h = 1)
  
  rolling_predictions[i] <- as.numeric(forecast_one_step$mean)
}


# 1. Assicurati che il ciclo 'for' per rolling_predictions sia stato eseguito
# ... (il codice del ciclo for va qui) ...

# 2. Crea il DataFrame di confronto
comparison_df_rolling <- data.frame(
  Indice_Tempo = seq_along(test$log_UMCSENT), # Numero della riga nel test set
  Valore_Effettivo = test$log_UMCSENT,        # I valori reali osservati
  Previsione_Rolling = rolling_predictions,    # I valori previsti a 1-passo
  Errore = test$log_UMCSENT - rolling_predictions # Calcolo dell'errore (Effettivo - Previsto)
)

# 3. Stampa il DataFrame di confronto
print(comparison_df_rolling)


# 3. Grafico di Confronto (Usando ggplot2, più robusto)
comparison_df_rolling <- data.frame(
  Time = seq_along(test$log_UMCSENT),
  Actual = test$log_UMCSENT,
  Predicted = rolling_predictions
)

library(ggplot2)
library(forecast)
library(zoo) # Necessario per la conversione del formato temporale

# --- 1. PREPARAZIONE DEI DATI E INDICI CORRETTI ---

# La serie completa deve essere di classe 'ts' (time series)
full_series <- data_filtered$log_UMCSENT
full_time_index <- time(full_series)

n_train <- length(train$log_UMCSENT)
n_test <- length(test$log_UMCSENT)
n_total <- n_train + n_test

# Converti l'indice numerico frazionario in un oggetto Data (fondamentale per ggplot)
if (frequency(full_series) %in% c(4, 12)) {
  # Per dati mensili (12) o trimestrali (4), usiamo as.Date(as.yearmon/yearqtr)
  date_index <- as.Date(as.yearmon(full_time_index))
  date_format <- "%Y-%m" # Formato Anno-Mese per l'etichetta
  date_break <- "1 year" # Intervallo di un anno tra le etichette
} else {
  # Per dati annuali (1) o frequenze non standard, usiamo l'indice numerico
  date_index <- full_time_index
  date_format <- "%Y"
  date_break <- "2" # Usato solo come placeholder, non influenza scale_x_continuous
}

# --- 2. CREAZIONE DEI DATAFRAME PER PLOTTING ---

# DataFrame per i Valori Effettivi (Training + Test)
df_actual <- data.frame(
  Time = date_index,
  Valore = c(train$log_UMCSENT, test$log_UMCSENT),
  Tipo_Serie = factor(c(rep("Training Set", n_train), rep("Test Set Effettivo", n_test)))
)

# DataFrame per le Previsioni Rolling (Solo Test)
# Correzione dell'indice: [dal punto n_train + 1] fino al punto [n_total]
df_predicted <- data.frame(
  Time = date_index[(n_train + 1):n_total],
  Valore = rolling_predictions,
  Tipo_Serie = factor(rep("Previsioni Rolling (1-step)", n_test))
)

# Unisci i due DataFrame
df_plot <- rbind(df_actual, df_predicted)

# Calcola la posizione della linea di separazione
vline_position_time <- full_time_index[n_train] + (1 / frequency(full_series) / 2)

# Converti la posizione della linea nel formato Time corretto
if (frequency(full_series) %in% c(4, 12)) {
  vline_position_date <- as.Date(as.yearmon(vline_position_time))
} else {
  vline_position_date <- vline_position_time
}


# --- 3. GENERAZIONE DEL GRAFICO CON DATE E SEPARATORE ---
p <- ggplot(df_plot, aes(x = Time, y = Valore, color = Tipo_Serie)) +
  geom_line(data = df_plot[df_plot$Tipo_Serie != "Previsioni Rolling (1-step)", ], 
            size = 1.1) +
  geom_line(data = df_plot[df_plot$Tipo_Serie == "Previsioni Rolling (1-step)", ], 
            size = 1, linetype = "dashed") +
  geom_vline(xintercept = vline_position_date,
             linetype = "dotted", 
             color = "black", 
             linewidth = 0.8) +
  labs(
    title = "Confronto Modello ARIMA Rolling: Training vs. Previsioni di Test",
    y = "log(UMCSENT)",
    x = "Tempo"
  ) +
  scale_color_manual(values = c("Training Set" = "darkgreen", 
                                "Test Set Effettivo" = "blue", 
                                "Previsioni Rolling (1-step)" = "red")) +
  theme_minimal() +
  guides(color = guide_legend(title = "Serie"))

# Aggiungi la formattazione dell'asse X (dove risolviamo l'errore)
if (frequency(full_series) %in% c(4, 12)) {
  # Per dati con formati data espliciti (mesi/trimestri)
  p <- p + scale_x_date(date_breaks = date_break, date_labels = date_format)
} else {
  # Per dati annuali (o numerici semplici) - Rimosso il calcolo problematico di 'seq.default'
  # Lasciamo che ggplot scelga gli intervalli di tempo, che saranno anni semplici
  p <- p + scale_x_continuous() 
}

print(p)


# RMSE
# 1. Calcola gli errori (residui)
errors <- test$log_UMCSENT - rolling_predictions

# 2. Calcola gli errori al quadrato
squared_errors <- errors^2

# 3. Calcola la media degli errori al quadrato (MSE)
mean_squared_error <- mean(squared_errors)

# 4. Calcola la radice quadrata (RMSE)
rmse_value <- sqrt(mean_squared_error)

# Stampa il risultato
print(paste("L'RMSE del modello ARIMA(0,1,0) è:", round(rmse_value, 6)))
