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

# ==============================================================================
# Read the data
# ==============================================================================
dm_data <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_data.csv")

delta <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/Model/Sliding Window/Delta_Results.csv")
decay <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/Model/Sliding Window/Decay_Results.csv")
igt_summary <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary_IGT.csv")
dm_summary_overall <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary_overall.csv")

# Possible levels: ('Nature', 'Urban', 'Control') or ('Urban', 'Nature', 'Control')
delta$Condition <- factor(delta$Condition, levels = c('Nature', 'Urban', 'Control'))
decay$Condition <- factor(decay$Condition, levels = c('Nature', 'Urban', 'Control'))
dm_data$Condition <- factor(dm_data$Condition, levels = c('Nature', 'Urban', 'Control')) 
igt_summary$Condition <- factor(igt_summary$Condition, levels = c('Nature', 'Urban', 'Control'))
dm_summary_overall$Condition <- factor(dm_summary_overall$Condition, levels = c('Nature', 'Urban', 'Control'))


delta_nature <- delta %>%
  filter(Condition=='Nature')

delta_igt <- delta %>%
  filter(task_id=='2')

igt <- dm_data %>%
  filter(Task=='IGT')

sgt <- dm_data %>%
  filter(Task=='SGT')

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

model <- glmer(BestOption ~ Condition + Block + (1|Subnum), family='binomial', data = igt)
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

sgt_2nd <- dm_summary_overall %>%
  filter(Order == 'SGT_IGT')


model <- glm(BestOption ~ Condition + (1|Subnum), data = igt_sgt)
summary(model)
plot(allEffects(model))

model <- lmer(BestOption ~ Condition + (1|Subnum), data = sgt_igt)
summary(model)
plot(allEffects(model))

model <- glm(BestOption_IGT ~ Condition + BestOption_SGT, data = sgt_2nd)
summary(model)
plot(allEffects(model))

model <- lmer(HighFreqOption ~ Condition + (1|Subnum), data = dm_data)
summary(model)
plot(allEffects(model))

