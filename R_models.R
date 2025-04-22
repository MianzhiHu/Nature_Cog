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
delta <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/Model/Sliding Window/Delta_Results.csv")
decay <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/Model/Sliding Window/Decay_Results.csv")
igt <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_summary_IGT.csv")
delta$Condition <- factor(delta$Condition, levels = c('Nature', 'Urban', 'Control'))
decay$Condition <- factor(decay$Condition, levels = c('Nature', 'Urban', 'Control'))
igt$Condition <- factor(igt$Condition, levels = c('Nature', 'Urban', 'Control'))
delta_nature <- delta %>%
  filter(Condition=='Nature')

delta_igt <- delta %>%
  filter(task_id=='2')

# ==============================================================================
# Generalized Additive Mixed Models
# ==============================================================================
m <- gam(alpha ~ Condition + s(window_id, by=Condition) + s(Subnum, bs="re"),
         data = delta)

plot(m)
summary(m)

# ==============================================================================
# Linear Mixed-Effects Models
# ==============================================================================
mixed_effect <- lmer(t ~ Condition * poly(window_id, 3) + (1 + window_id|Subnum),
              data = delta)

summary(mixed_effect)
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
model <- lmer(BestOption ~ Condition + Block + (1|Subnum), data = igt)
summary(model)

model <- lmer(BestOption ~ Condition * Block + naturalness + disorderliness
              + aesthetic + (1|Subnum), data = igt)
summary(model)

model <- lmer(alpha ~ Condition * window_id + (1|Subnum), data = delta_igt)
summary(model)

model <- lmer(t ~ Condition + window_id + (1|Subnum), data = delta_igt)
summary(model)

