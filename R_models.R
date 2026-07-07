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
library(emmeans)

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

E1_pls <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/PLS_Data/PLS_Sem_E1.csv")


# Possible levels: ('Nature', 'Urban', 'Control') or ('Urban', 'Nature', 'Control')
dm_data$Condition <- factor(dm_data$Condition, levels = c('Control', 'Nature', 'Urban')) 
dm_data$Condition <- factor(dm_data$Condition, levels = c('Nature', 'Urban', 'Control')) 
dm_data$Task <- factor(dm_data$Task, levels = c(1, 2), labels = c("First", "Second"))
dm_summary$Condition <- factor(dm_summary$Condition, levels = c('Nature', 'Urban', 'Control'))

dm_summary_modeled$Condition <- factor(dm_summary_modeled$Condition, levels = c('Control', 'Nature', 'Urban'))
dm_summary_modeled$Task <- factor(dm_summary_modeled$Task, levels = c(1, 2), labels = c("First", "Second"))

wsls_data$Condition <- factor(wsls_data$Condition, levels = c('Nature', 'Urban', 'Control'))
wsls_data$Task <- factor(wsls_data$Task, levels = c(1, 2), labels = c("First", "Second"))
wsls_data$exploration <- factor(wsls_data$exploration, levels = c(0, 1), labels = c('exploitation', 'exploration'))
wsls_data$EV_rank <- factor(wsls_data$EV_rank, levels = c('1', '2', '3', '4'))

E2_wsls_data$Condition <- factor(E2_wsls_data$Condition, levels = c('Nature', 'Urban', 'Control'))
E2_wsls_data$Task <- factor(E2_wsls_data$Task, levels = c(1, 2), labels = c("First", "Second"))
E2_wsls_data$exploration <- factor(E2_wsls_data$exploration, levels = c(0, 1), labels = c('exploitation', 'exploration'))
E2_wsls_data$EV_rank <- factor(E2_wsls_data$EV_rank, levels = c('1', '2', '3', '4'))

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
# Win-Stay-Lose-Shift Behavior
# ==============================================================================
wsls_2 <- wsls_data %>%
  filter(Task == 'Second')
wsls_exploration <- wsls_data %>%
  filter(exploration == 'exploration')


# Possible levels: ('Nature', 'Urban', 'Control') or ('Urban', 'Nature', 'Control') or ('Control', 'Nature', 'Urban')
wsls_data$Condition <- factor(wsls_data$Condition, levels = c('Control', 'Nature', 'Urban'))

reward_model <- lmer(Reward ~ Condition * Task + (1|Subnum), data = wsls_data)
summary(reward_model)
plot(allEffects(reward_model))

optimal_model <- glmer(BestChoice ~ Condition * Task + (1|Subnum), family=binomial, data = wsls_data)
summary(optimal_model)
plot(allEffects(optimal_model))

value_dis_model <- lmer(value_gap ~ Condition * Task + (1|Subnum), data = wsls_data)
summary(value_dis_model)
plot(allEffects(value_dis_model))

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

ex_model <- glmer(LoseShift ~ Condition + (1|Subnum), family=binomial, data = E2_wsls_data)
summary(ex_model)
plot(allEffects(ex_model))

ex2_model <- glmer(rank_2 ~ Condition * Task + (1|Subnum), family=binomial, data = wsls_exploration)
summary(ex2_model)
plot(allEffects(ex2_model))
emm <- as.data.frame(confint(emmeans(ex2_model, ~ Condition * Task)))
write.csv(emm, "rank_2_emmeans.csv", row.names = FALSE)

ex2_model <- glmer(rank_2 ~ Condition + (1|Subnum), family=binomial, data = wsls_150_E2)
summary(ex2_model)
plot(allEffects(ex2_model))



rank_model <- clmm(EV_rank ~ Condition * Task + (1 | Subnum), data = wsls_data, link = "logit")
summary(rank_model)
plot(ggpredict(rank_model, terms = c("Condition", "Task")))

rank_model <- clmm(EV_rank ~ Condition * Task + (1 | Subnum), data = wsls_exploration, link = "logit")
summary(rank_model)
plot(ggpredict(rank_model, terms = c("Condition", "Task")))

rank_model <- clmm(EV_rank ~ Condition + (1 | Subnum), data = E2_wsls_data, link = "logit")
summary(rank_model)
plot(ggpredict(rank_model, terms = c("Condition", "Task")))

ev_model <- lmer(EV_history ~ Condition * Task + (1|Subnum), data = wsls_data)
summary(ev_model)
plot(allEffects(ev_model))

ev_model <- lmer(EV_history ~ Condition * Task + (1|Subnum), data = wsls_exploration)
summary(ev_model)
plot(allEffects(ev_model))
emm <- as.data.frame(confint(emmeans(ev_model, ~ Condition * Task)))
write.csv(emm, "EV_history_emmeans.csv", row.names = FALSE)

ex_model <- glm(EV_rank_difference ~ (sky+grass+plant+water+fence+path+river+bench+pole+building+
                  tree+earth+rock+streetlight+wall+signboard+sidewalk+railing+road+person+mountain) + Condition, data = E1_pls)
summary(ex_model)
plot(allEffects(ex_model))

# ==============================================================================
# E2 Behavioral Analysis
# ==============================================================================
E2_wsls_exploration <- E2_wsls_data %>%
  filter(exploration == 'exploration')

model <- lmer(Reward ~ Condition + (1|Subnum), data = E2_wsls_data)
summary(model)
plot(allEffects(model))

model <- glmer(BestChoice ~ Condition + (1|Subnum), family='binomial', data = E2_wsls_data)
summary(model)
plot(allEffects(model))

value_dis_model <- lmer(value_gap ~ Condition + (1|Subnum), data = E2_wsls_data)
summary(value_dis_model)
plot(allEffects(value_dis_model))

switch_model <- glmer(Switch ~ Condition + (1|Subnum), family=binomial, data = E2_wsls_data)
summary(switch_model)
plot(allEffects(switch_model))

ws_model <- glmer(WinStay ~ Condition + (1|Subnum), family=binomial, data = E2_wsls_data)
summary(ws_model)
plot(allEffects(ws_model))

ls_model <- glmer(LoseShift ~ Condition + (1|Subnum), family=binomial, data = E2_wsls_data)
summary(ls_model)
plot(allEffects(ls_model))

ex_model <- glmer(exploration ~ Condition + (1|Subnum), family=binomial, data = E2_wsls_data)
summary(ex_model)
plot(allEffects(ex_model))

ex2_model <- glmer(rank_2 ~ Condition + (1|Subnum), family=binomial, data = E2_wsls_exploration)
summary(ex2_model)
plot(allEffects(ex2_model))
emm <- as.data.frame(confint(emmeans(ex2_model, ~ Condition)))
write.csv(emm, "E2_rank_2_emmeans.csv", row.names = FALSE)

ev_model <- lmer(EV_history ~ Condition + (1|Subnum), data = E2_wsls_data)
summary(ev_model)
plot(allEffects(ev_model))

ev_model <- lmer(EV_history ~ Condition + (1|Subnum), data = E2_wsls_exploration)
summary(ev_model)
plot(allEffects(ev_model))
emm <- as.data.frame(confint(emmeans(ev_model, ~ Condition)))
write.csv(emm, "E2_EV_history_emmeans.csv", row.names = FALSE)

ex_model <- glmer(exploration ~ (sky+grass+plant+water+fence+path+river+bench+
                                   pole+building+tree+earth+rock+streetlight+
                                   wall+signboard+sidewalk+railing+road+person+
                                   mountain) * Condition + (1|Subnum), family=binomial, data = E2_wsls_data)
summary(ex_model)
plot(allEffects(ex_model))


# ==============================================================================
# E1 Participant-Level Semantic Difference Analysis
# ==============================================================================
# The saved E1 PLS data has one row per participant and z-scored task-difference
# outcomes. Because there is only one row per participant, a participant random
# intercept is not estimable here; these are participant-level linear models.

e1_semantic_features <- c(
  "sky", "grass", "plant", "water", "fence", "path", "river", "bench",
  "pole", "building", "tree", "earth", "rock", "streetlight", "wall",
  "signboard", "sidewalk", "railing", "road", "person", "mountain"
)

e1_difference_outcomes <- c(
  "Reward_difference",
  "BestChoice_difference",
  "value_gap_difference",
  "Switch_difference",
  "WinStay_difference",
  "LoseShift_difference",
  "t_difference",
  "dis_sd_difference",
  "noise_sd_difference",
  "decay_difference",
  "decay_center_difference",
  "Exploration_Rate_difference",
  "rank_2_exploration_rate_difference",
  "EV_history_exploration_difference"
)

e1_semantic_output_dir <- "C:/Users/zuire/PycharmProjects/Nature_Cog/analysis_outputs"
dir.create(e1_semantic_output_dir, showWarnings = FALSE, recursive = TRUE)

E1_semantic_difference_data <- read.csv(
  "C:/Users/zuire/PycharmProjects/Nature_Cog/data/PLS_Data/PLS_Sem_E1.csv"
)
E1_semantic_difference_data$Condition <- factor(
  E1_semantic_difference_data$Condition,
  levels = c("Nature", "Urban")
)

standardize_e1_vector <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x)) || is.na(sd(x, na.rm = TRUE)) || sd(x, na.rm = TRUE) == 0) {
    return(rep(NA_real_, length(x)))
  }
  as.numeric(scale(x))
}

fit_e1_difference_model <- function(data, outcome, interaction = FALSE) {
  rhs <- if (interaction) {
    "feature_z * Condition"
  } else {
    "feature_z + Condition"
  }
  lm(as.formula(paste0(outcome, " ~ ", rhs)), data = data)
}

extract_e1_difference_coefficients <- function(model, outcome, feature_name,
                                               model_type, n_obs, n_subjects) {
  coef_table <- as.data.frame(coef(summary(model)))
  coef_table$term <- rownames(coef_table)
  rownames(coef_table) <- NULL

  bind_cols(
    tibble(
      level = "participant_difference",
      outcome = outcome,
      feature = feature_name,
      model_type = model_type,
      model = "lm",
      n_obs = n_obs,
      n_subjects = n_subjects,
      error = NA_character_
    ),
    as_tibble(coef_table)
  )
}

run_e1_semantic_difference_grid <- function(data, outcomes, features) {
  results <- list()
  row_id <- 1

  for (outcome in outcomes) {
    base_data <- data %>%
      filter(!is.na(.data[[outcome]]), !is.na(Condition), !is.na(Subnum))

    for (feature_name in features) {
      model_data <- base_data %>%
        filter(!is.na(.data[[feature_name]])) %>%
        mutate(
          feature_z = standardize_e1_vector(.data[[feature_name]]),
          Condition = droplevels(Condition)
        ) %>%
        filter(!is.na(feature_z))

      if (nrow(model_data) == 0 || nlevels(model_data$Condition) < 2) {
        next
      }

      n_obs <- nrow(model_data)
      n_subjects <- dplyr::n_distinct(model_data$Subnum)

      for (model_type in c("additive", "interaction")) {
        model <- tryCatch(
          fit_e1_difference_model(
            model_data,
            outcome,
            interaction = model_type == "interaction"
          ),
          error = function(e) e
        )

        if (inherits(model, "error")) {
          results[[row_id]] <- tibble(
            level = "participant_difference",
            outcome = outcome,
            feature = feature_name,
            model_type = model_type,
            model = "lm",
            n_obs = n_obs,
            n_subjects = n_subjects,
            error = model$message,
            term = NA_character_
          )
        } else {
          results[[row_id]] <- extract_e1_difference_coefficients(
            model,
            outcome,
            feature_name,
            model_type,
            n_obs,
            n_subjects
          )
        }

        row_id <- row_id + 1
      }
    }
  }

  bind_rows(results)
}

E1_semantic_difference_results <- run_e1_semantic_difference_grid(
  E1_semantic_difference_data,
  e1_difference_outcomes,
  e1_semantic_features
)

write.csv(
  E1_semantic_difference_results,
  file.path(e1_semantic_output_dir, "e1_semantic_difference_model_coefficients_R.csv"),
  row.names = FALSE
)

cat("\nE1 semantic difference analysis complete.\n")
cat("Saved coefficient table to e1_semantic_difference_model_coefficients_R.csv\n")


# ==============================================================================
# E2 Trial-Wise Semantic Feature Analysis
# ==============================================================================
# This section tests each semantic feature one at a time and saves the fixed-effect
# coefficient table directly from summary(lmer/glmer).

semantic_features <- c(
  "sky", "grass", "plant", "water", "fence", "path", "river", "bench",
  "pole", "building", "tree", "earth", "rock", "streetlight", "wall",
  "signboard", "sidewalk", "railing", "road", "person", "mountain"
)

trial_outcomes <- tibble::tribble(
  ~outcome,      ~family,     ~subset,
  "Reward",     "gaussian",  "all",
  "BestChoice", "binomial",  "all",
  "value_gap",  "gaussian",  "all",
  "Switch",     "binomial",  "all",
  "WinStay",    "binomial",  "all",
  "LoseShift",  "binomial",  "all",
  "exploration","binomial",  "all",
  "rank_2",     "binomial",  "exploration_only",
  "EV_history", "gaussian",  "exploration_only"
)

parameter_outcomes <- tibble::tribble(
  ~outcome,        ~family,    ~subset,
  "t",             "gaussian", "participant_mean_semantics",
  "dis_sd",        "gaussian", "participant_mean_semantics",
  "noise_sd",      "gaussian", "participant_mean_semantics",
  "decay",         "gaussian", "participant_mean_semantics",
  "decay_center",  "gaussian", "participant_mean_semantics"
)

semantic_output_dir <- "C:/Users/zuire/PycharmProjects/Nature_Cog/data"
dir.create(semantic_output_dir, showWarnings = FALSE, recursive = TRUE)

E2_semantic_trial_data <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/E2_dm_switch.csv")
E2_semantic_param_data <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/E2_dm_summary_modeled.csv")

E2_semantic_trial_data$Condition <- factor(
  E2_semantic_trial_data$Condition,
  levels = c("Nature", "Urban", "Control")
)
E2_semantic_param_data$Condition <- factor(
  E2_semantic_param_data$Condition,
  levels = c("Nature", "Urban", "Control")
)

standardize_vector <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x)) || is.na(sd(x, na.rm = TRUE)) || sd(x, na.rm = TRUE) == 0) {
    return(rep(NA_real_, length(x)))
  }
  as.numeric(scale(x))
}

make_trial_subset <- function(data, outcome, subset_name) {
  model_data <- data
  if (subset_name == "exploration_only") {
    model_data <- model_data %>% filter(exploration == 1)
  }
  model_data %>%
    filter(!is.na(.data[[outcome]]), !is.na(Condition), !is.na(Subnum))
}

make_parameter_data <- function(trial_data, parameter_data) {
  semantic_means <- trial_data %>%
    group_by(Subnum, Condition) %>%
    summarise(across(all_of(semantic_features), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

  parameter_data %>%
    left_join(semantic_means, by = c("Subnum", "Condition")) %>%
    mutate(Condition = factor(Condition, levels = c("Nature", "Urban", "Control")))
}

fit_semantic_model <- function(data, outcome, family_name, interaction = FALSE, mixed = TRUE) {
  rhs <- if (interaction) {
    "feature_z * Condition"
  } else {
    "feature_z + Condition"
  }

  if (mixed) {
    formula_text <- paste0(outcome, " ~ ", rhs, " + Trial + (1|Subnum)")
    if (family_name == "binomial") {
      glmer(
        as.formula(formula_text),
        data = data,
        family = binomial
      )
    } else {
      lmer(as.formula(formula_text), data = data)
    }
  } else {
    formula_text <- paste0(outcome, " ~ ", rhs)
    lm(as.formula(formula_text), data = data)
  }
}

extract_model_coefficients <- function(model, level_name, outcome, family_name, subset_name,
                                       feature_name, model_type, n_obs, n_subjects,
                                       model_name) {
  coef_table <- as.data.frame(coef(summary(model)))
  coef_table$term <- rownames(coef_table)
  rownames(coef_table) <- NULL

  bind_cols(
    tibble(
      level = level_name,
      outcome = outcome,
      family = family_name,
      subset = subset_name,
      feature = feature_name,
      model_type = model_type,
      model = model_name,
      n_obs = n_obs,
      n_subjects = n_subjects,
      error = NA_character_
    ),
    as_tibble(coef_table)
  )
}

run_semantic_feature_grid <- function(data, outcomes, level_name, mixed = TRUE) {
  results <- list()
  row_id <- 1

  for (outcome_i in seq_len(nrow(outcomes))) {
    outcome <- outcomes$outcome[outcome_i]
    family_name <- outcomes$family[outcome_i]
    subset_name <- outcomes$subset[outcome_i]

    base_data <- if (level_name == "trial") {
      make_trial_subset(data, outcome, subset_name)
    } else {
      data %>% filter(!is.na(.data[[outcome]]), !is.na(Condition), !is.na(Subnum))
    }

    for (feature_name in semantic_features) {
      model_data <- base_data %>%
        filter(!is.na(.data[[feature_name]])) %>%
        mutate(
          feature_z = standardize_vector(.data[[feature_name]]),
          Condition = droplevels(Condition)
        ) %>%
        filter(!is.na(feature_z))

      if (nrow(model_data) == 0 || nlevels(model_data$Condition) < 2) {
        next
      }

      n_obs <- nrow(model_data)
      n_subjects <- dplyr::n_distinct(model_data$Subnum)
      model_name <- ifelse(mixed, ifelse(family_name == "binomial", "glmer", "lmer"), "lm")

      for (model_type in c("additive", "interaction")) {
        model <- tryCatch(
          fit_semantic_model(
            model_data,
            outcome,
            family_name,
            interaction = model_type == "interaction",
            mixed = mixed
          ),
          error = function(e) e
        )

        if (inherits(model, "error")) {
          results[[row_id]] <- tibble(
            level = level_name,
            outcome = outcome,
            family = family_name,
            subset = subset_name,
            feature = feature_name,
            model_type = model_type,
            model = model_name,
            n_obs = n_obs,
            n_subjects = n_subjects,
            error = model$message,
            term = NA_character_
          )
        } else {
          results[[row_id]] <- extract_model_coefficients(
            model,
            level_name,
            outcome,
            family_name,
            subset_name,
            feature_name,
            model_type,
            n_obs,
            n_subjects,
            model_name
          )
        }

        row_id <- row_id + 1
      }
    }
  }

  bind_rows(results)
}

E2_semantic_parameter_data <- make_parameter_data(
  E2_semantic_trial_data,
  E2_semantic_param_data
)

trial_semantic_results <- run_semantic_feature_grid(
  E2_semantic_trial_data,
  trial_outcomes,
  level_name = "trial",
  mixed = TRUE
)

parameter_semantic_results <- run_semantic_feature_grid(
  E2_semantic_parameter_data,
  parameter_outcomes,
  level_name = "participant_parameter",
  mixed = FALSE
)

E2_semantic_model_results <- bind_rows(
  trial_semantic_results,
  parameter_semantic_results
)

write.csv(
  E2_semantic_model_results,
  file.path(semantic_output_dir, "e2_semantic_feature_model_coefficients_R.csv"),
  row.names = FALSE
)

cat("\nE2 semantic feature analysis complete.\n")
cat("Saved fixed-effect coefficient table to e2_semantic_feature_model_coefficients_R.csv\n")

