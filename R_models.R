library(ggplot2)
library(lme4)
library(lmerTest)
library(effects)
library(ggplot2)
library(dplyr)
library(mgcv)
library(sjPlot)
library(segmented)
library(survival)
library(changepoint)
library(seminr)
library(randomForest)
library(randomForestExplainer)
library(tidyr)
library(purrr)
library(tibble)
library(readr)
library(ordinal)
library(ggeffects)

# ==============================================================================
# Read the data
# ==============================================================================
dm_data <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/E1_dm_data.csv")
dm_summary <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary.csv")
dm_summary_wide <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/E1_dm_summary_task_wide.csv")
dm_summary_modeled <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary_modeled.csv")
dm_composite <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/agg_condition_value_counts.csv")

wsls_data <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_switch.csv")
E2_wsls_data <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/E2_dm_switch.csv")
exploration <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/E1_exploration_data.csv")
E1_behavioral_mw <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/E1_behavioral_moving_window.csv")


# Possible levels: ('Nature', 'Urban', 'Control') or ('Urban', 'Nature', 'Control')
dm_data$Condition <- factor(dm_data$Condition, levels = c('Control', 'Nature', 'Urban')) 
dm_data$Condition <- factor(dm_data$Condition, levels = c('Nature', 'Urban', 'Control')) 
dm_data$Task <- factor(dm_data$Task, levels = c(1, 2), labels = c("First", "Second"))
dm_summary$Condition <- factor(dm_summary$Condition, levels = c('Nature', 'Urban', 'Control'))

dm_summary_modeled$Condition <- factor(dm_summary_modeled$Condition, levels = c('Control', 'Nature', 'Urban'))
dm_summary_modeled$Task <- factor(dm_summary_modeled$Task, levels = c(1, 2), labels = c("First", "Second"))

wsls_data$Condition <- factor(wsls_data$Condition, levels = c('Control', 'Nature', 'Urban'))
wsls_data$Task <- factor(wsls_data$Task, levels = c(1, 2), labels = c("First", "Second"))
wsls_data$exploration <- factor(wsls_data$exploration, levels = c(0, 1), labels = c('exploitation', 'exploration'))
wsls_data$EV_rank <- factor(wsls_data$EV_rank, levels = c('1st', '2nd', '3rd', '4th'))

E2_wsls_data$Condition <- factor(E2_wsls_data$Condition, levels = c('Nature', 'Urban', 'Control'))
E2_wsls_data$Task <- factor(E2_wsls_data$Task, levels = c(1, 2), labels = c("First", "Second"))
E2_wsls_data$exploration <- factor(E2_wsls_data$exploration, levels = c(0, 1), labels = c('exploitation', 'exploration'))
# E2_wsls_data$EV_rank <- factor(E2_wsls_data$EV_rank, levels = c('1st', '2nd', '3rd', '4th'))

dm_composite$agg_Condition <- factor(dm_composite$agg_Condition, levels = c('Mid Composite Score', 'High Composite Score', 'Low Composite Score'))
dm_composite$Task <- factor(dm_composite$Task, levels = c(1, 2), labels = c("First", "Second"))
dm_composite$exploration <- factor(dm_composite$exploration, levels = c(0, 1), labels = c('exploitation', 'exploration'))


exploration$Condition <- factor(exploration$Condition, levels = c('Control', 'Nature', 'Urban'))
exploration$Task <- factor(exploration$Task, levels = c(1, 2), labels = c("First", "Second"))
exploration$exploration <- factor(exploration$exploration, levels = c("exploitation", "exploration"))

E1_behavioral_mw$Condition <- factor(E1_behavioral_mw$Condition, levels = c('Control', 'Nature', 'Urban'))
E1_behavioral_mw$Task <- factor(E1_behavioral_mw$Task, levels = c(1, 2), labels = c("First", "Second"))

dm_summary_modeled_wide <- dm_summary_modeled %>%
  pivot_wider(
    id_cols = c(Subnum, Condition),
    names_from = Task,
    values_from = Exploration,
    names_prefix = "Task_"
  )

# ==============================================================================
# Generalized Additive Mixed Models
# ==============================================================================
m <- gam(
  Exploration ~ Condition + s(WindowStart) + s(WindowStart, by = Condition) + s(Subnum, bs = "re"),
  data = E1_behavioral_mw_2,
  method = "REML"
)

plot(m)
summary(m)
p <- plot_model(m,type  = "pred", 
                terms = c("window_id [all]", "Condition"))
p + geom_vline(xintercept = 91, linetype = "dotted")

# ==============================================================================
# Linear Mixed-Effects Models
# ==============================================================================
mixed_effect <- lmer(alpha ~ Condition * poly(window_id, 3) + (1 + window_id|Subnum),
              data = delta)

summary(mixed_effect)
anova(mixed_effect)
p <- plot_model(mixed_effect,type  = "pred", 
           terms = c("window_id [all]", "Condition"))
p + geom_vline(xintercept = 91, linetype = "dotted")

# Basic behavioral;
model <- glm(HighFreqOption_IGT ~ Condition + HighFreqOption_SGT,
                     data = dm_summary_overall)

summary(model)
plot(allEffects(model))


anova(mixed_effect)
p <- plot_model(mixed_effect,type  = "pred", 
                terms = c("window_id [all]", "Condition"))
p + geom_vline(xintercept = 91, linetype = "dotted")

# ==============================================================================
# General Linear Model for Differences
# ==============================================================================
diff <- delta %>%
  filter(window_id==92)

model <- glm(alpha_diff ~ naturalness + disorderliness + aesthetic + Condition,
              data=diff)
summary(model)
plot(allEffects(model))

# ==============================================================================
# General Linear Model for images
# ==============================================================================
stimuli_info <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/stimuli/stimuli_influence.csv")
nature <- stimuli_info %>%
  filter(Category == "Nature")
urban <- stimuli_info %>%
  filter(Category == "Urban")
inf <- glm(Influence ~ poly(Semantic_PC1, 2), data=stimuli_info)
summary(inf)
plot(allEffects(inf))

inf <- glm(Influence ~ sky+grass+plant+water+sea+fence+path+river+bench+pole+
           building+tree+earth+rock+streetlight+ashcan+table+wall+chair+signboard+
             stairs+pot+sculpture+sidewalk+railing+road+person+mountain+lake+floor
           +car+traffic.light, data=urban)
summary(inf)
plot(allEffects(inf))

inf <- glm(Influence ~ (Hue+SDHue+Bright+SDBright+Saturaton+SDSat+Contrast+
             Dissimilarity+Homogeneity+Energy+Correlation+MeanTexture+SDTexture+
             Entropy+EdgeCount+CornerMean+CornerSD+CornerCount+ContourMeanLength+
             ContourSDLength+ContourMeanArea+ContourSDArea+ContourCount+AsymmetryV+
             AsymmetryH+KPMeanSize+KPSDSize+KPMeanStrength+KPSDStrength+KPMeanAngle+
             KPSDAngle+KPCount)*Category, data=stimuli_info)
summary(inf)
plot(allEffects(inf))

# ==============================================================================
# Behavioral Analysis
# ==============================================================================
model <- glm(Reward ~ Exploration_Rate * Condition * Task, data = dm_summary_modeled)
summary(model)

model <- glmer(BestChoice ~ Condition * Task + Block + (1|Subnum), family='binomial', data = dm_data)
summary(model)
plot(allEffects(model))

model <- lmer(Reward ~ Condition * Task + Block + (1|Subnum), data = dm_data)
summary(model)
plot(allEffects(model))

model <- lmer(BestOption ~ Condition * Block + naturalness + disorderliness
              + aesthetic + (1|Subnum), data = igt_summary)
summary(model)

model <- lmer(alpha ~ Condition + window_id + (1|Subnum), data = delta_igt)
summary(model)

model <- lmer(t ~ Condition * Task + (1|Subnum), data = dm_summary_modeled)
summary(model)

model <- glm(Task_Second ~ Condition * Task_First, data = dm_summary_modeled_wide)
summary(model)

model <- glmer(exploration ~ Condition * Task + (1|Subnum), family=binomial, data = exploration)
summary(model)
plot(allEffects(model))

# ==============================================================================
# Win-Stay-Lose-Shift Behavior
# ==============================================================================
wsls_2 <- wsls_data %>%
  filter(Task == 'Second')


reward_model <- lmer(Reward ~ Condition * Task + (1|Subnum), data = wsls_data)
summary(reward_model)
plot(allEffects(reward_model))

optimal_model <- glmer(BestChoice ~ Condition * Task + (1|Subnum), family=binomial, data = wsls_data)
summary(optimal_model)
plot(allEffects(optimal_model))

switch_model <- glmer(Switch ~ Condition * Task + (1|Subnum), family=binomial, data = wsls_data)
summary(switch_model)
plot(allEffects(switch_model))

switch_model <- glmer(exploration ~ agg_Condition * Task  + (1|Subnum), family=binomial, data = dm_composite)
summary(switch_model)
plot(allEffects(switch_model))

ws_model <- glmer(WinStay ~ Condition * Task + (1|Subnum), family=binomial, data = wsls_data)
summary(ws_model)
plot(allEffects(ws_model))

ls_model <- glmer(LoseShift ~ Condition * Task + (1|Subnum), family=binomial, data = wsls_data)
summary(ls_model)
plot(allEffects(ls_model))

ex_model <- glmer(exploration ~ Condition * Task + (1|Subnum), family=binomial, data = wsls_data)
summary(ex_model)
plot(allEffects(ex_model))

ex_model <- glmer(exploration ~ Condition * poly(Trial, 2) + (1|Subnum), family=binomial, data = E2_wsls_data)
summary(ex_model)
plot(allEffects(ex_model))

rank_model <- clmm(EV_rank ~ Condition * Task + (1 | Subnum), data = wsls_data, link = "logit")
summary(rank_model)
plot(ggpredict(rank_model, terms = c("Condition", "Task")))

rank_model <- clmm(EV_rank ~ Condition + (1 | Subnum), data = E2_wsls_data, link = "logit")
summary(rank_model)
plot(ggpredict(rank_model, terms = c("Condition", "Task")))

ev_model <- lmer(EV_history ~ Condition * Task + (1|Subnum), data = wsls_data)
summary(ev_model)
plot(allEffects(ev_model))

# ==============================================================================
# Behavioral Moving Window
# ==============================================================================
E1_behavioral_mw_2 <- E1_behavioral_mw %>%
  filter(Task=='Second')
wsls_data_2 <- wsls_data %>%
  filter(Task=='Second')

bmw_e1 <- lmer(Exploration ~ Condition * poly(WindowStart, 2) + (1 + poly(WindowStart, 2) | Subnum), data = E1_behavioral_mw_2)
summary(bmw_e1)
plot(allEffects(bmw_e1))

bmw_e1 <- glmer(Switch ~ Condition * poly(Trial, 2) + (1|Subnum), family=binomial, data = wsls_data_2)
summary(bmw_e1)
plot(allEffects(bmw_e1))

# ==============================================================================
# Extract E1 residuals
# ==============================================================================
# ID columns
id_col <- "Subnum"
task_col <- "Task"
task1_label <- 'First'
task2_label <- 'Second'
linear_behavior_vars <- c(
  "BestChoice",
  "Reward",
  "Switch",
  "WinStay",
  "LoseShift",
  "exploration",
  "t",
  "dis_sd",
  "noise_sd",
  "decay",
  "decay_center"
)

quadratic_behavior_vars <- c(
  "BestChoice",
  "Reward",
  "Switch",
  "WinStay",
  "LoseShift",
  "Exploration"
)

# Define functions to extract residuals
extract_linear_residuals <- function(data, var, task1, task2) {
  
  col_task1 <- paste0(var, "_", task1)
  col_task2 <- paste0(var, "_", task2)
  
  # Filter data
  tmp <- data %>%
    dplyr::select(all_of(id_col), all_of(col_task1), all_of(col_task2)) %>%
    dplyr::filter(
      !is.na(.data[[col_task1]]),
      !is.na(.data[[col_task2]])
    )
  
  # Fit Task2 ~ Task1
  formula_txt <- paste(col_task2, "~", col_task1)
  model <- lm(as.formula(formula_txt), data = tmp)
  
  # Add residual columns
  tmp[[paste0(var, "_resid")]] <- resid(model)
  
  cat("\n=========================\n")
  cat("Behavior index:", var, "\n")
  print(summary(model))
  
  tmp %>%
    dplyr::select(
      all_of(id_col),
      all_of(paste0(var, "_resid"))
    )
}

extract_mw_quadratic_residuals <- function(data, var, id_col = "Subnum") {
  
  formula_txt <- paste0(var, " ~ poly(WindowStart, 2) + 
                        (1 + poly(WindowStart, 2) | ", id_col,")")
  
  cat("\n=========================\n")
  cat("Fitting variable:", var, "\n")
  cat("Formula:", formula_txt, "\n")
  
  model <- lmer(as.formula(formula_txt), data = data)
  print(summary(model))
  
  # Extract random effects
  re_df <- ranef(model)[[id_col]] %>%
    as.data.frame() %>%
    rownames_to_column(var = id_col)
  
  # Rename columns to include variable name
  names(re_df) <- c(
    id_col,
    paste0(var, "_2nd_Intercept"),
    paste0(var, "_2nd_Linear"),
    paste0(var, "_2nd_Quadratic")
  )
  
  # Match ID type to original data
  if (is.numeric(data[[id_col]])) {
    re_df[[id_col]] <- as.numeric(re_df[[id_col]])
  } else {
    re_df[[id_col]] <- as.character(re_df[[id_col]])
  }
  
  re_df
}

# Load data
linear_residual_df <- wsls_data
linear_residual_df <- linear_residual_df %>%
  mutate(exploration = ifelse(exploration == "exploration", 1, 0))

summary_fun <- \(x) mean(x, na.rm = TRUE)

# Generate summary as grouped by task
linear_residual_df_summary <- linear_residual_df %>%
  group_by(.data[[id_col]], .data[[task_col]]) %>%
  summarise(
    across(all_of(linear_behavior_vars), summary_fun, .names = "{.col}"),
    .groups = "drop"
  )

# Check result
print(linear_residual_df_summary)

linear_residual_df_summary_wide <- linear_residual_df_summary %>%
  pivot_wider(
    names_from = all_of(task_col),
    values_from = all_of(linear_behavior_vars),
    names_sep = "_"
  )

print(linear_residual_df_summary_wide)

# Start extracting
linear_residual_list <- purrr::map(
  linear_behavior_vars,
  ~ extract_linear_residuals(
    data = linear_residual_df_summary_wide,
    var = .x,
    task1 = task1_label,
    task2 = task2_label
  )
)

linear_residuals <- reduce(linear_residual_list, full_join, by = id_col)

# Now quadratic residuals
quadratic_residuals_list <- purrr::map(
  quadratic_behavior_vars,
  ~ extract_mw_quadratic_residuals(
    data = E1_behavioral_mw_2,
    var = .x,
    id_col = "Subnum"
  )
)

# Combine all random-effect outputs by Subnum
quadratic_residuals <- purrr::reduce(quadratic_residuals_list, dplyr::full_join, by = "Subnum")

# Merge residuals
residuals_all <- linear_residuals %>%
  left_join(quadratic_residuals, by = "Subnum")
write_csv(residuals_all, "./data/behavior_residuals.csv")

# # ==============================================================================
# PLS SEM
E1_pls <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/PLS_Data/PLS_Sem_E1.csv")
model <- glm(Switch_resid ~ (sky+grass+plant+water+fence+path+river+bench+
                                         pole+building+tree+earth+rock+streetlight+wall+
                                         signboard+sidewalk+railing+road+person+mountain)*Condition, data=E1_pls)
summary(model)
plot(allEffects(model))

# Example for one variable (you would repeat for all or use a loop/dplyr)
E1_pls <- E1_pls %>%
  group_by(Condition) %>%
  mutate(grass_mean = mean(grass),
         grass_dev = grass - mean(grass))

# New Model
model_decomp <- glm(Switch_resid ~ Condition + grass_mean + grass_dev, data = E1_pls)
summary(model_decomp)
plot(allEffects(model_decomp))

E1_pls$Condition <- factor(E1_pls$Condition, levels = c('Nature', 'Urban'))
E1_pls <- E1_pls %>%
  mutate(Cond01 = ifelse(Condition == "Nature", 1, 0))
E1_pls_nat <- E1_pls %>%
  filter(Condition == 'Nature')
E1_pls_urb <- E1_pls %>%
  filter(Condition == 'Urban')

mm <- constructs(
  # composite("Visual", c('Hue', 'Bright', 'Saturaton', 'SDhue', 'SDsat', 'Sdbright',
  #                       'Entropy', 'SED', 'NSED')),
  composite("Rating", c('naturalness', 'disorderliness',
                        'aesthetic', 'familiarity', 'engagement', 'fascination', 'mystery',
                        'imagability', 'control')),
  
  composite("Visual", c('sky', 'grass', 'plant', 'water', 'fence', 'path', 'river', 'bench', 'pole', 'building',
                        'tree', 'earth', 'rock', 'streetlight', 'wall', 'signboard', 'sidewalk', 'railing', 'road',
                        'person', 'mountain')),

  composite("Behavior", c('BestChoice_resid', 'Reward_resid',
                          'Switch_resid', 'WinStay_resid', 'LoseShift_resid')),
  # composite('Behavior', single_item('BestOption_SGT')),
  # composite('Behavior', c('BestOption_SGT', 'HighMagOption_SGT')),
  composite("Cond", single_item("Cond01"))
  # composite("Params", c("t_Diff_z","alpha_Diff_z","shape_Diff_z","la_Diff_z"))
  # composite("Params", c("t_SGT","alpha_SGT","shape_SGT","la_SGT"))
)

sm <- relationships(
  paths(from = "Cond",  to = "Visual"),
  paths(from = "Visual",  to = c("Rating", "Behavior")),
  paths(from = "Rating",  to = "Behavior")
)

# sm <- relationships(
#   paths(from = "Cond",  to = "Visual"),
#   paths(from = "Visual",  to = c("Behavior", "Params")),
#   # paths(from = "Rating",  to = c("Params", "Behavior")),
#   paths(from = "Params",  to = "Behavior")
# )


model <- estimate_pls(E1_pls, measurement_model = mm, structural_model = sm)
summary(model)

boot_mobi_pls <- bootstrap_model(seminr_model = model,
                                 nboot = 1000,
                                 cores = 32)
summary(boot_mobi_pls)
plot(boot_mobi_pls, title = "Bootstrapped Model")

# Extract construct scores
scores <- as.data.frame(model$construct_scores)

lm1 <- lm(Behavior ~ Visual + Cond, data = scores)
summary(lm1)
plot(allEffects(lm1))

# # ==============================================================================
# # Change-point analysis
# # ==============================================================================
# # Function to detect change-point for each participant:
# detect_change <- function(data_sub){
#   # You can adjust method and penalty as needed
#   cpt <- cpt.meanvar(data_sub$alpha, method = "PELT", penalty = "BIC")
#   return(cpts(cpt)[1]) # First detected change-point
# }
# 
# # Run change-point detection per participant (only Task 2)
# change_points <- delta %>%
#   # filter(task_id == 2) %>%
#   group_by(Subnum, Condition) %>%
#   summarise(change_point_trial = detect_change(cur_data()),
#             .groups = "drop")
# 
# change_points
# 
# cpt <- cpts(cpt.meanvar(delta$alpha, method = "PELT", penalty = "BIC"))
