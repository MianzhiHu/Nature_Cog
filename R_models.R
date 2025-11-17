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

# ==============================================================================
# Read the data
# ==============================================================================
dm_data <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_data_summary.csv")
dm_summary <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary.csv")
dm_summary_wide <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary_task_wide.csv")
dm_summary_modeled <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary_modeled.csv")
dm_summary_modeled_wide <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary_modeled_wide.csv")
deck_summary <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/deck_summary.csv")
IGT_SGT_summary <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/IGT_SGT_summary.csv")

delta <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/Model/Sliding Window/Delta_Results.csv")
decay <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/Model/Sliding Window/Decay_Results.csv")


# Possible levels: ('Nature', 'Urban', 'Control') or ('Urban', 'Nature', 'Control')
dm_data$Condition <- factor(dm_data$Condition, levels = c('Nature', 'Urban', 'Control')) 
dm_summary$Condition <- factor(dm_summary$Condition, levels = c('Nature', 'Urban', 'Control'))
dm_summary_modeled$Condition <- factor(dm_summary_modeled$Condition, levels = c('Nature', 'Urban', 'Control'))
dm_summary_modeled_wide$Condition <- factor(dm_summary_modeled_wide$Condition, levels = c('Nature', 'Urban', 'Control'))
dm_summary_wide$Condition <- factor(dm_summary_wide$Condition, levels = c('Nature', 'Urban', 'Control'))
IGT_SGT_summary$Condition <- factor(IGT_SGT_summary$Condition, levels = c('Nature', 'Baseline', 'Urban', 'Control'))
deck_summary$Condition <- factor(deck_summary$Condition, levels = c('Nature', 'Urban', 'Control'))
deck_summary$keyResponse <- factor(deck_summary$keyResponse, levels = c(1, 2, 3, 4), labels = c('A', 'B', 'C', 'D'))

dm_summary$TaskCode <- factor(dm_summary$Condition)
dm_data$TaskCode <- factor(dm_data$TaskCode) 

delta$Condition <- factor(delta$Condition, levels = c('Nature', 'Urban', 'Control'))
decay$Condition <- factor(decay$Condition, levels = c('Nature', 'Urban', 'Control'))

igt_sgt_wide <- dm_summary_wide %>%
  filter(Order == 'IGT_SGT')

delta_nature <- delta %>%
  filter(Condition=='Nature')

delta_igt <- delta %>%
  filter(task_id=='2')

igt <- dm_data %>%
  filter(Task=='IGT')

sgt <- dm_data %>%
  filter(Task=='SGT')

deck_summary_igt_sgt <- deck_summary %>%
  filter(Task=='SGT' & Order == 'IGT_SGT')

# ==============================================================================
# Generalized Additive Mixed Models
# ==============================================================================
m <- gam(alpha ~ Condition + s(window_id, by=Condition) + s(Subnum, bs="re"),
         data = delta)

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
# Change-point analysis
# ==============================================================================
# Function to detect change-point for each participant:
detect_change <- function(data_sub){
  # You can adjust method and penalty as needed
  cpt <- cpt.meanvar(data_sub$alpha, method = "PELT", penalty = "BIC")
  return(cpts(cpt)[1]) # First detected change-point
}

# Run change-point detection per participant (only Task 2)
change_points <- delta %>%
  # filter(task_id == 2) %>%
  group_by(Subnum, Condition) %>%
  summarise(change_point_trial = detect_change(cur_data()),
            .groups = "drop")

change_points

cpt <- cpts(cpt.meanvar(delta$alpha, method = "PELT", penalty = "BIC"))

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
# Behavioral Analysis
# ==============================================================================
model <- lmer(BestOption ~ Condition + Block + (1|Subnum), data = igt_summary)
summary(model)

model <- glmer(BestOption ~ Condition + TaskCode + Block + (1|Subnum), family='binomial', data = igt)
summary(model)

model <- lmer(BestOption ~ Condition * Block + naturalness + disorderliness
              + aesthetic + (1|Subnum), data = igt_summary)
summary(model)

model <- lmer(alpha ~ Condition + window_id + (1|Subnum), data = delta_igt)
summary(model)

model <- lmer(t ~ Condition + window_id + (1|Subnum), data = delta_igt)
summary(model)

# ==============================================================================
# Random
igt_sgt <- dm_data %>%
  filter(Task == 'SGT') %>%
  filter(Order == 'IGT_SGT')

sgt_igt <- dm_data %>%
  filter(Task == 'IGT') %>%
  filter(Order == 'SGT_IGT')

sgt_2nd <- dm_summary_wide %>%
  filter(Order == 'IGT_SGT')

igt_2nd <- dm_summary %>%
  filter(Order == 'SGT_IGT')

dm_summary_modeled <- dm_summary_modeled %>%
  filter(Order == 'IGT_SGT')

dm_summary_modeled_wide <- dm_summary_modeled_wide %>%
  filter(Order == 'IGT_SGT')


model <- lmer(BestOption_z ~ Condition * Task * TaskCode + Order  + (1|Subnum), data = dm_summary)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ Condition * TaskCode + (1|Subnum), family=binomial, data = sgt)
summary(model)
plot(allEffects(model))

model <- glm(BestOption_z_IGT ~ Condition + BestOption_z_SGT, data = igt_2nd)
summary(model)
plot(allEffects(model))

model <- glm(BestOption_z_SGT ~ Condition + BestOption_z_IGT, data = sgt_2nd)
summary(model)
plot(allEffects(model))

model <- lmer(HighFreqOption ~ Condition + (1|Subnum), data = dm_data)
summary(model)
plot(allEffects(model))

model <- glm(BestOption_z_Diff ~ Condition, data = igt_sgt_wide)
summary(model)
plot(allEffects(model))

model <- glmer(BestOption ~ Condition + (1|Subnum), family=binomial, data = IGT_SGT_summary)
summary(model)
plot(allEffects(model))


model <- glm(BestOption_z_Diff ~ Condition * Order, data = dm_summary_wide)
summary(model)
plot(allEffects(model))

model <- glm(ChoiceRate_z ~ Condition * keyResponse, data = deck_summary_igt_sgt)
summary(model)
plot(allEffects(model))

model <- glm(t ~ Condition + Task, data = dm_summary_modeled)
summary(model)
plot(allEffects(model))

model <- glm(la_Diff_z ~ Condition, data = dm_summary_modeled_wide)
summary(model)
plot(allEffects(model))



# PLS SEM
igt_sgt_pls <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/PLS_Data/PLS_Sem_IGT_SGT.csv")
igt_sgt_pls$Condition <- factor(igt_sgt_pls$Condition, levels = c('Nature', 'Urban'))
igt_sgt_pls <- igt_sgt_pls %>%
  mutate(Cond01 = ifelse(Condition == "Nature", 1, 0))
igt_sgt_pls_nat <- igt_sgt_pls %>%
  filter(Condition == 'Nature')
igt_sgt_pls_urb <- igt_sgt_pls %>%
  filter(Condition == 'Urban')

mm <- constructs(
  composite("Visual", c('Hue', 'Bright', 'Saturaton', 'SDhue', 'SDsat', 'Sdbright',
                        'Entropy', 'SED', 'NSED')),
  # composite("Rating", c("naturalness", "disorderliness", "aesthetic")),
  
  composite("Behavior", c('BestOption_Optim_z_Diff')),
  # composite('Behavior', single_item('BestOption_SGT')),
  # composite('Behavior', c('BestOption_SGT', 'HighMagOption_SGT')),
  composite("Cond", single_item("Cond01"))
  # composite("Params", c("t_Diff_z","alpha_Diff_z","shape_Diff_z","la_Diff_z"))
  # composite("Params", c("t_SGT","alpha_SGT","shape_SGT","la_SGT"))
)

sm <- relationships(
  paths(from = "Cond",  to = "Visual"),
  paths(from = "Visual",  to = c("Behavior"))
  # paths(from = "Rating",  to = "Behavior")
)

# sm <- relationships(
#   paths(from = "Cond",  to = "Visual"),
#   paths(from = "Visual",  to = c("Behavior", "Params")),
#   # paths(from = "Rating",  to = c("Params", "Behavior")),
#   paths(from = "Params",  to = "Behavior")
# )


model <- estimate_pls(igt_sgt_pls, measurement_model = mm, structural_model = sm)
summary(model)

boot_mobi_pls <- bootstrap_model(seminr_model = model,
                                 nboot = 10000,
                                 cores = 32)
summary(boot_mobi_pls)
plot(boot_mobi_pls, title = "Bootstrapped Model")

# Extract construct scores
scores <- as.data.frame(model$construct_scores)

lm1 <- lm(Behavior ~ Visual + Cond, data = scores)
summary(lm1)
plot(allEffects(lm1))
