library(ggplot2)
library(rlang)
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
dm_summary_modeled <- read.csv("D:/PycharmProjects/Nature_Cog/data/dm_summary_modeled.csv")
dm_composite <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/agg_condition_value_counts.csv")
E2_dm_summary_modeled <- read.csv("D:/PycharmProjects/Nature_Cog/data/E2_dm_summary_modeled.csv")

wsls_data <- read.csv("C:/Users/zuire/PycharmProjects/Nature_Cog/data/dm_switch.csv")
E2_wsls_data <- read.csv("D:/PycharmProjects/Nature_Cog/data/E2_dm_switch.csv")
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
# Possible levels: ('Nature', 'Urban', 'Control') or ('Control', 'Nature', 'Urban')
E2_wsls_data$Condition <- factor(E2_wsls_data$Condition, levels = c('Control', 'Nature', 'Urban'))
E2_wsls_data$Task <- factor(E2_wsls_data$Task, levels = c(1, 2), labels = c("First", "Second"))
E2_wsls_data$exploration <- factor(E2_wsls_data$exploration, levels = c(0, 1), labels = c('exploitation', 'exploration'))
E2_wsls_data$EV_rank <- factor(E2_wsls_data$EV_rank, levels = c('1', '2', '3', '4'))

E2_wsls_exploration <- E2_wsls_data %>%
  filter(exploration == 'exploration')

reward_model <- lmer(Reward ~ Condition + (1|Subnum), data = E2_wsls_data)
summary(reward_model)
plot(allEffects(reward_model))

optimal_model <- glmer(BestChoice ~ Condition + (1|Subnum), family='binomial', data = E2_wsls_data)
summary(optimal_model)
plot(allEffects(optimal_model))

value_dis_model <- lmer(value_gap ~ Condition + (1|Subnum), data = E2_wsls_data)
summary(value_dis_model)
plot(allEffects(value_dis_model))

switch_model <- glmer(Switch ~ Condition + (1|Subnum), family='binomial', data = E2_wsls_data)
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

# E2 Parameter Analysis
E2_dm_summary_modeled$Condition <- 
  factor(E2_dm_summary_modeled$Condition, levels = c('Nature', 'Urban', 'Control'))

t_model <- glm(t ~ Condition, data = E2_dm_summary_modeled)
summary(t_model)
plot(allEffects(t_model))

dis_sd_model <- glm(dis_sd ~ Condition, data = E2_dm_summary_modeled)
summary(dis_sd_model)
plot(allEffects(dis_sd_model))

noise_sd_model <- glm(noise_sd ~ Condition, data = E2_dm_summary_modeled)
summary(noise_sd_model)
plot(allEffects(noise_sd_model))

decay_model <- glm(decay ~ Condition, data = E2_dm_summary_modeled)
summary(decay_model)
plot(allEffects(decay_model))

decay_center_model <- glm(decay_center ~ Condition, data = E2_dm_summary_modeled)
summary(decay_center_model)
plot(allEffects(decay_center_model))




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
  "Exploration_Rate_difference",
  "rank_2_exploration_rate_difference",
  "EV_history_exploration_difference"
)

e1_semantic_output_dir <- "analysis_outputs"
dir.create(e1_semantic_output_dir, showWarnings = FALSE, recursive = TRUE)

E1_semantic_difference_data <- read.csv(
  file.path("data", "PLS_Data", "PLS_Sem_E1.csv")
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
# E2 Semantic Feature Analysis
# ==============================================================================
# Trial-level behavioral outcomes use lmer/glmer with a participant random
# intercept. Participant-level model parameters use Gaussian glm models.
# Trial number is intentionally not included as a covariate.

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
  ~outcome,       ~family,    ~subset,
  "t",            "gaussian", "participant_mean_semantics",
  "dis_sd",       "gaussian", "participant_mean_semantics",
  "noise_sd",     "gaussian", "participant_mean_semantics",
  "decay",        "gaussian", "participant_mean_semantics",
  "decay_center", "gaussian", "participant_mean_semantics"
)

semantic_output_dir <- "data"
dir.create(semantic_output_dir, showWarnings = FALSE, recursive = TRUE)

E2_semantic_trial_data <- read.csv(file.path("data", "E2_dm_switch.csv"))

E2_semantic_trial_data$Condition <- factor(
  E2_semantic_trial_data$Condition,
  levels = c("Nature", "Urban", "Control")
)

E2_semantic_means <- E2_semantic_trial_data %>%
  mutate(Condition = as.character(Condition)) %>%
  group_by(Subnum, Condition) %>%
  summarise(
    across(
      all_of(semantic_features),
      ~ if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE)
    ),
    .groups = "drop"
  )

E2_semantic_parameter_data <- read.csv(
  file.path("data", "E2_dm_summary_modeled.csv")
) %>%
  mutate(Condition = as.character(Condition)) %>%
  left_join(E2_semantic_means, by = c("Subnum", "Condition")) %>%
  mutate(
    Condition = factor(
      Condition,
      levels = c("Nature", "Urban", "Control")
    )
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

fit_semantic_model <- function(data, outcome, family_name, interaction = FALSE) {
  rhs <- if (interaction) {
    "feature_z * Condition"
  } else {
    "feature_z + Condition"
  }

  formula_text <- paste0(outcome, " ~ ", rhs, " + (1|Subnum) + (1 | image_name)")
  if (family_name == "binomial") {
    glmer(
      as.formula(formula_text),
      data = data,
      family = binomial
    )
  } else {
    lmer(as.formula(formula_text), data = data)
  }
}

fit_parameter_model <- function(data, outcome, interaction = FALSE) {
  rhs <- if (interaction) {
    "feature_z * Condition"
  } else {
    "feature_z + Condition"
  }

  glm(
    as.formula(paste0(outcome, " ~ ", rhs)),
    data = data,
    family = gaussian()
  )
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

run_semantic_feature_grid <- function(data, outcomes) {
  results <- list()
  row_id <- 1

  for (outcome_i in seq_len(nrow(outcomes))) {
    outcome <- outcomes$outcome[outcome_i]
    family_name <- outcomes$family[outcome_i]
    subset_name <- outcomes$subset[outcome_i]

    base_data <- make_trial_subset(data, outcome, subset_name)

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
      model_name <- ifelse(family_name == "binomial", "glmer", "lmer")

      for (model_type in c("additive", "interaction")) {
        model <- tryCatch(
          fit_semantic_model(
            model_data,
            outcome,
            family_name,
            interaction = model_type == "interaction"
          ),
          error = function(e) e
        )

        if (inherits(model, "error")) {
          results[[row_id]] <- tibble(
            level = "trial",
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
            "trial",
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

run_parameter_feature_grid <- function(data, outcomes) {
  results <- list()
  row_id <- 1

  for (outcome_i in seq_len(nrow(outcomes))) {
    outcome <- outcomes$outcome[outcome_i]
    family_name <- outcomes$family[outcome_i]
    subset_name <- outcomes$subset[outcome_i]

    base_data <- data %>%
      filter(!is.na(.data[[outcome]]), !is.na(Condition), !is.na(Subnum))

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

      for (model_type in c("additive", "interaction")) {
        model <- tryCatch(
          fit_parameter_model(
            model_data,
            outcome,
            interaction = model_type == "interaction"
          ),
          error = function(e) e
        )

        if (inherits(model, "error")) {
          results[[row_id]] <- tibble(
            level = "participant_parameter",
            outcome = outcome,
            family = family_name,
            subset = subset_name,
            feature = feature_name,
            model_type = model_type,
            model = "glm",
            n_obs = n_obs,
            n_subjects = n_subjects,
            error = model$message,
            term = NA_character_
          )
        } else {
          results[[row_id]] <- extract_model_coefficients(
            model,
            "participant_parameter",
            outcome,
            family_name,
            subset_name,
            feature_name,
            model_type,
            n_obs,
            n_subjects,
            "glm"
          )
        }

        row_id <- row_id + 1
      }
    }
  }

  bind_rows(results)
}

trial_semantic_results <- run_semantic_feature_grid(
  E2_semantic_trial_data,
  trial_outcomes
)

parameter_semantic_results <- run_parameter_feature_grid(
  E2_semantic_parameter_data,
  parameter_outcomes
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



# ==============================================================================
# E2 Joint Semantic Feature Analysis: 9 behavioral outcomes x 2 model types
# ==============================================================================
# All 21 standardized semantic features are entered together. The additive model
# estimates their mutually adjusted associations; the interaction model also
# estimates a separate feature slope for Urban relative to Nature. Trial number
# and the five participant-level model parameters are intentionally excluded.
# No multiple-comparison correction is applied in this output.

E2_joint_semantic_features <- c(
  "sky", "grass", "plant", "water", "fence", "path", "river", "bench",
  "pole", "building", "tree", "earth", "rock", "streetlight", "wall",
  "signboard", "sidewalk", "railing", "road", "person", "mountain"
)

E2_joint_trial_outcomes <- tibble::tribble(
  ~outcome,      ~family,     ~subset,
  "Reward",      "gaussian",  "all",
  "BestChoice",  "binomial",  "all",
  "value_gap",   "gaussian",  "all",
  "Switch",      "binomial",  "all",
  "WinStay",     "binomial",  "all",
  "LoseShift",   "binomial",  "all",
  "exploration", "binomial",  "all",
  "rank_2",      "binomial",  "exploration_only",
  "EV_history",  "gaussian",  "exploration_only"
)

E2_joint_semantic_data <- read.csv(file.path("data", "E2_dm_switch.csv")) %>%
  mutate(
    Condition = factor(Condition, levels = c("Nature", "Urban", "Control"))
  )

E2_joint_z_features <- paste0(E2_joint_semantic_features, "_z")
E2_joint_additive_rhs <- paste(
  c(E2_joint_z_features, "Condition"),
  collapse = " + "
)
E2_joint_interaction_rhs <- paste0(
  "(", paste(E2_joint_z_features, collapse = " + "), ") * Condition"
)

E2_joint_output_file <- file.path(
  "data",
  "e2_joint_semantic_behavior_model_coefficients_R.csv"
)

E2_joint_results <- list()
E2_joint_row_id <- 1

for (outcome_i in seq_len(nrow(E2_joint_trial_outcomes))) {
  outcome <- E2_joint_trial_outcomes$outcome[outcome_i]
  family_name <- E2_joint_trial_outcomes$family[outcome_i]
  subset_name <- E2_joint_trial_outcomes$subset[outcome_i]

  model_data <- E2_joint_semantic_data
  if (subset_name == "exploration_only") {
    model_data <- model_data %>% filter(exploration == 1)
  }

  model_data <- model_data %>%
    filter(
      !is.na(.data[[outcome]]),
      !is.na(Condition),
      !is.na(Subnum),
      if_all(all_of(E2_joint_semantic_features), ~ !is.na(.x))
    ) %>%
    mutate(
      across(
        all_of(E2_joint_semantic_features),
        ~ as.numeric(scale(as.numeric(.x))),
        .names = "{.col}_z"
      ),
      Condition = droplevels(Condition)
    ) %>%
    filter(if_all(all_of(E2_joint_z_features), ~ !is.na(.x)))

  n_obs <- nrow(model_data)
  n_subjects <- dplyr::n_distinct(model_data$Subnum)
  n_images <- dplyr::n_distinct(model_data$image_name)

  for (model_type in c("additive", "interaction")) {
    rhs <- if (model_type == "interaction") {
      E2_joint_interaction_rhs
    } else {
      E2_joint_additive_rhs
    }
    formula_text <- paste0(outcome, " ~ ", rhs, " + (1 | Subnum) + (1 | image_name)")
    model_name <- ifelse(family_name == "binomial", "glmer", "lmer")
    optimizer_name <- ifelse(family_name == "binomial", "bobyqa", "nloptwrap")
    model_warnings <- character()

    model <- tryCatch(
      withCallingHandlers(
        {
          if (family_name == "binomial") {
            glmer(
              as.formula(formula_text),
              data = model_data,
              family = binomial,
              nAGQ = 0,
              control = glmerControl(
                optimizer = "bobyqa",
                optCtrl = list(maxfun = 200000)
              )
            )
          } else {
            lmer(as.formula(formula_text), data = model_data)
          }
        },
        warning = function(w) {
          model_warnings <<- c(model_warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) e
    )

    if (inherits(model, "error")) {
      E2_joint_results[[E2_joint_row_id]] <- tibble(
        level = "trial",
        outcome = outcome,
        family = family_name,
        subset = subset_name,
        feature = "all_semantic_features",
        model_type = model_type,
        model = model_name,
        formula = formula_text,
        optimizer = optimizer_name,
        nAGQ = ifelse(family_name == "binomial", 0, NA_real_),
        n_obs = n_obs,
        n_subjects = n_subjects,
        n_images = n_images,
        converged = FALSE,
        singular = NA,
        AIC = NA_real_,
        BIC = NA_real_,
        logLik = NA_real_,
        deviance = NA_real_,
        convergence_message = NA_character_,
        warning = ifelse(
          length(model_warnings) == 0,
          NA_character_,
          paste(unique(model_warnings), collapse = " | ")
        ),
        error = conditionMessage(model),
        term = NA_character_
      )
    } else {
      coefficient_table <- as.data.frame(coef(summary(model)))
      coefficient_table$term <- rownames(coefficient_table)
      rownames(coefficient_table) <- NULL

      convergence_messages <- model@optinfo$conv$lme4$messages
      converged <- is.null(convergence_messages)
      model_singular <- lme4::isSingular(model, tol = 1e-4)
      model_aic <- AIC(model)
      model_bic <- BIC(model)
      model_loglik <- as.numeric(logLik(model))
      model_deviance <- if (family_name == "binomial") {
        deviance(model)
      } else {
        lme4::REMLcrit(model)
      }

      E2_joint_results[[E2_joint_row_id]] <- bind_cols(
        tibble(
          level = "trial",
          outcome = outcome,
          family = family_name,
          subset = subset_name,
          feature = "all_semantic_features",
          model_type = model_type,
          model = model_name,
          formula = formula_text,
          optimizer = optimizer_name,
          nAGQ = ifelse(family_name == "binomial", 0, NA_real_),
          n_obs = n_obs,
          n_subjects = n_subjects,
          n_images = n_images,
          converged = converged,
          singular = model_singular,
          AIC = model_aic,
          BIC = model_bic,
          logLik = model_loglik,
          deviance = model_deviance,
          convergence_message = ifelse(
            converged,
            NA_character_,
            paste(convergence_messages, collapse = " | ")
          ),
          warning = ifelse(
            length(model_warnings) == 0,
            NA_character_,
            paste(unique(model_warnings), collapse = " | ")
          ),
          error = NA_character_
        ),
        as_tibble(coefficient_table)
      )
    }

    E2_joint_row_id <- E2_joint_row_id + 1

    write.csv(
      bind_rows(E2_joint_results),
      E2_joint_output_file,
      row.names = FALSE
    )

    cat(
      sprintf(
        "Completed E2 joint model %d/18: %s (%s)\n",
        E2_joint_row_id - 1,
        outcome,
        model_type
      )
    )
  }
}

cat("\nE2 joint semantic behavior analysis complete.\n")
cat("Saved coefficient and model-detail table to ", E2_joint_output_file, "\n", sep = "")


# ==============================================================================
# E2 Joint Rating Analysis: 9 behavioral outcomes x 2 model types
# ==============================================================================
# All nine standardized image ratings are entered together. Ratings are complete
# for Nature, Urban, and Control, so all three conditions are retained with Nature
# as the reference level. Trial number and participant-level model parameters are
# intentionally excluded. No multiple-comparison correction is applied here.

E2_joint_rating_features <- c(
  "naturalness", "disorderliness", "aesthetic", "familiarity",
  "engagement", "fascination", "mystery", "imagability", "control"
)

E2_joint_rating_outcomes <- tibble::tribble(
  ~outcome,      ~family,     ~subset,
  "Reward",      "gaussian",  "all",
  "BestChoice",  "binomial",  "all",
  "value_gap",   "gaussian",  "all",
  "Switch",      "binomial",  "all",
  "WinStay",     "binomial",  "all",
  "LoseShift",   "binomial",  "all",
  "exploration", "binomial",  "all",
  "rank_2",      "binomial",  "exploration_only",
  "EV_history",  "gaussian",  "exploration_only"
)

E2_joint_rating_data <- read.csv(file.path("data", "E2_dm_switch.csv")) %>%
  mutate(
    Condition = factor(Condition, levels = c("Nature", "Urban", "Control"))
  )

E2_joint_z_ratings <- paste0(E2_joint_rating_features, "_z")
E2_joint_rating_additive_rhs <- paste(
  c(E2_joint_z_ratings, "Condition"),
  collapse = " + "
)
E2_joint_rating_interaction_rhs <- paste0(
  "(", paste(E2_joint_z_ratings, collapse = " + "), ") * Condition"
)

E2_joint_rating_output_file <- file.path(
  "data",
  "e2_joint_rating_behavior_model_coefficients_R.csv"
)

E2_joint_rating_results <- list()
E2_joint_rating_row_id <- 1

for (outcome_i in seq_len(nrow(E2_joint_rating_outcomes))) {
  outcome <- E2_joint_rating_outcomes$outcome[outcome_i]
  family_name <- E2_joint_rating_outcomes$family[outcome_i]
  subset_name <- E2_joint_rating_outcomes$subset[outcome_i]

  model_data <- E2_joint_rating_data
  if (subset_name == "exploration_only") {
    model_data <- model_data %>% filter(exploration == 1)
  }

  model_data <- model_data %>%
    filter(
      !is.na(.data[[outcome]]),
      !is.na(Condition),
      !is.na(Subnum),
      if_all(all_of(E2_joint_rating_features), ~ !is.na(.x))
    ) %>%
    mutate(
      across(
        all_of(E2_joint_rating_features),
        ~ as.numeric(scale(as.numeric(.x))),
        .names = "{.col}_z"
      ),
      Condition = droplevels(Condition)
    ) %>%
    filter(if_all(all_of(E2_joint_z_ratings), ~ !is.na(.x)))

  n_obs <- nrow(model_data)
  n_subjects <- dplyr::n_distinct(model_data$Subnum)
  n_images <- dplyr::n_distinct(model_data$image_name)

  for (model_type in c("additive", "interaction")) {
    rhs <- if (model_type == "interaction") {
      E2_joint_rating_interaction_rhs
    } else {
      E2_joint_rating_additive_rhs
    }
    formula_text <- paste0(outcome, " ~ ", rhs, " + (1 | Subnum) + (1 | image_name)")
    model_name <- ifelse(family_name == "binomial", "glmer", "lmer")
    optimizer_name <- ifelse(family_name == "binomial", "bobyqa", "nloptwrap")
    model_warnings <- character()

    model <- tryCatch(
      withCallingHandlers(
        {
          if (family_name == "binomial") {
            glmer(
              as.formula(formula_text),
              data = model_data,
              family = binomial,
              nAGQ = 0,
              control = glmerControl(
                optimizer = "bobyqa",
                optCtrl = list(maxfun = 200000)
              )
            )
          } else {
            lmer(as.formula(formula_text), data = model_data)
          }
        },
        warning = function(w) {
          model_warnings <<- c(model_warnings, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) e
    )

    if (inherits(model, "error")) {
      E2_joint_rating_results[[E2_joint_rating_row_id]] <- tibble(
        level = "trial",
        outcome = outcome,
        family = family_name,
        subset = subset_name,
        feature = "all_ratings",
        model_type = model_type,
        model = model_name,
        formula = formula_text,
        optimizer = optimizer_name,
        nAGQ = ifelse(family_name == "binomial", 0, NA_real_),
        n_obs = n_obs,
        n_subjects = n_subjects,
        n_images = n_images,
        converged = FALSE,
        singular = NA,
        AIC = NA_real_,
        BIC = NA_real_,
        logLik = NA_real_,
        deviance = NA_real_,
        convergence_message = NA_character_,
        warning = ifelse(
          length(model_warnings) == 0,
          NA_character_,
          paste(unique(model_warnings), collapse = " | ")
        ),
        error = conditionMessage(model),
        term = NA_character_
      )
    } else {
      coefficient_table <- as.data.frame(coef(summary(model)))
      coefficient_table$term <- rownames(coefficient_table)
      rownames(coefficient_table) <- NULL

      convergence_messages <- model@optinfo$conv$lme4$messages
      converged <- is.null(convergence_messages)
      model_singular <- lme4::isSingular(model, tol = 1e-4)
      model_aic <- AIC(model)
      model_bic <- BIC(model)
      model_loglik <- as.numeric(logLik(model))
      model_deviance <- if (family_name == "binomial") {
        deviance(model)
      } else {
        lme4::REMLcrit(model)
      }

      E2_joint_rating_results[[E2_joint_rating_row_id]] <- bind_cols(
        tibble(
          level = "trial",
          outcome = outcome,
          family = family_name,
          subset = subset_name,
          feature = "all_ratings",
          model_type = model_type,
          model = model_name,
          formula = formula_text,
          optimizer = optimizer_name,
          nAGQ = ifelse(family_name == "binomial", 0, NA_real_),
          n_obs = n_obs,
          n_subjects = n_subjects,
          n_images = n_images,
          converged = converged,
          singular = model_singular,
          AIC = model_aic,
          BIC = model_bic,
          logLik = model_loglik,
          deviance = model_deviance,
          convergence_message = ifelse(
            converged,
            NA_character_,
            paste(convergence_messages, collapse = " | ")
          ),
          warning = ifelse(
            length(model_warnings) == 0,
            NA_character_,
            paste(unique(model_warnings), collapse = " | ")
          ),
          error = NA_character_
        ),
        as_tibble(coefficient_table)
      )
    }

    E2_joint_rating_row_id <- E2_joint_rating_row_id + 1

    write.csv(
      bind_rows(E2_joint_rating_results),
      E2_joint_rating_output_file,
      row.names = FALSE
    )

    cat(
      sprintf(
        "Completed E2 joint rating model %d/18: %s (%s)\n",
        E2_joint_rating_row_id - 1,
        outcome,
        model_type
      )
    )
  }
}

cat("\nE2 joint rating behavior analysis complete.\n")
cat(
  "Saved coefficient and model-detail table to ",
  E2_joint_rating_output_file,
  "\n",
  sep = ""
)


# ==============================================================================
# E1 Trial-wise Semantic Feature Analysis of Image Ratings
# ==============================================================================
# Nine trial-wise ratings are predicted from all 21 standardized semantic
# features. Additive and feature-by-Condition interaction models include crossed
# participant and image random intercepts. Control trials are excluded by
# complete-case filtering because their semantic features are unavailable.
# No multiple-comparison correction is applied in this output.

E1_trial_rating_semantic_features <- c(
  "sky", "grass", "plant", "water", "fence", "path", "river", "bench",
  "pole", "building", "tree", "earth", "rock", "streetlight", "wall",
  "signboard", "sidewalk", "railing", "road", "person", "mountain"
)

E1_trial_rating_outcomes <- c(
  "naturalness", "disorderliness", "aesthetic", "familiarity",
  "engagement", "fascination", "mystery", "imagability", "control"
)

E1_trial_rating_data <- read.csv(file.path("data", "E1_img_data.csv")) %>%
  mutate(
    Condition = factor(Condition, levels = c("Nature", "Urban", "Control"))
  )

E1_trial_rating_z_features <- paste0(
  E1_trial_rating_semantic_features,
  "_z"
)
E1_trial_rating_additive_rhs <- paste(
  c(E1_trial_rating_z_features, "Condition"),
  collapse = " + "
)
E1_trial_rating_interaction_rhs <- paste0(
  "(",
  paste(E1_trial_rating_z_features, collapse = " + "),
  ") * Condition"
)

E1_trial_rating_output_file <- file.path(
  "data",
  "e1_trialwise_semantic_rating_model_coefficients_R.csv"
)

E1_trial_rating_results <- list()
E1_trial_rating_row_id <- 1

for (outcome in E1_trial_rating_outcomes) {
  model_data <- E1_trial_rating_data %>%
    filter(
      !is.na(.data[[outcome]]),
      !is.na(Condition),
      !is.na(Subnum),
      !is.na(image_name),
      if_all(
        all_of(E1_trial_rating_semantic_features),
        ~ !is.na(.x)
      )
    ) %>%
    mutate(
      across(
        all_of(E1_trial_rating_semantic_features),
        ~ as.numeric(scale(as.numeric(.x))),
        .names = "{.col}_z"
      ),
      Condition = droplevels(Condition)
    ) %>%
    filter(
      if_all(
        all_of(E1_trial_rating_z_features),
        ~ !is.na(.x)
      )
    )

  n_obs <- nrow(model_data)
  n_subjects <- dplyr::n_distinct(model_data$Subnum)
  n_images <- dplyr::n_distinct(model_data$image_name)

  for (model_type in c("additive", "interaction")) {
    rhs <- if (model_type == "interaction") {
      E1_trial_rating_interaction_rhs
    } else {
      E1_trial_rating_additive_rhs
    }

    formula_text <- paste0(
      outcome,
      " ~ ",
      rhs,
      " + (1 | Subnum) + (1 | image_name)"
    )
    model_warnings <- character()

    model <- tryCatch(
      withCallingHandlers(
        lmer(
          as.formula(formula_text),
          data = model_data
        ),
        warning = function(w) {
          model_warnings <<- c(
            model_warnings,
            conditionMessage(w)
          )
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) e
    )

    if (inherits(model, "error")) {
      E1_trial_rating_results[[E1_trial_rating_row_id]] <- tibble(
        level = "trial_rating",
        outcome = outcome,
        family = "gaussian",
        subset = "all",
        feature = "all_semantic_features",
        model_type = model_type,
        model = "lmer",
        formula = formula_text,
        optimizer = "nloptwrap",
        nAGQ = NA_real_,
        n_obs = n_obs,
        n_subjects = n_subjects,
        n_images = n_images,
        converged = FALSE,
        singular = NA,
        AIC = NA_real_,
        BIC = NA_real_,
        logLik = NA_real_,
        deviance = NA_real_,
        convergence_message = NA_character_,
        warning = ifelse(
          length(model_warnings) == 0,
          NA_character_,
          paste(
            unique(model_warnings),
            collapse = " | "
          )
        ),
        error = conditionMessage(model),
        term = NA_character_
      )
    } else {
      coefficient_table <- as.data.frame(
        coef(summary(model))
      )
      coefficient_table$term <- rownames(
        coefficient_table
      )
      rownames(coefficient_table) <- NULL

      convergence_messages <- model@optinfo$conv$lme4$messages
      converged <- is.null(convergence_messages)
      model_singular <- lme4::isSingular(
        model,
        tol = 1e-4
      )
      model_aic <- AIC(model)
      model_bic <- BIC(model)
      model_loglik <- as.numeric(logLik(model))
      model_deviance <- lme4::REMLcrit(model)

      E1_trial_rating_results[[E1_trial_rating_row_id]] <- bind_cols(
        tibble(
          level = "trial_rating",
          outcome = outcome,
          family = "gaussian",
          subset = "all",
          feature = "all_semantic_features",
          model_type = model_type,
          model = "lmer",
          formula = formula_text,
          optimizer = "nloptwrap",
          nAGQ = NA_real_,
          n_obs = n_obs,
          n_subjects = n_subjects,
          n_images = n_images,
          converged = converged,
          singular = model_singular,
          AIC = model_aic,
          BIC = model_bic,
          logLik = model_loglik,
          deviance = model_deviance,
          convergence_message = ifelse(
            converged,
            NA_character_,
            paste(
              convergence_messages,
              collapse = " | "
            )
          ),
          warning = ifelse(
            length(model_warnings) == 0,
            NA_character_,
            paste(
              unique(model_warnings),
              collapse = " | "
            )
          ),
          error = NA_character_
        ),
        as_tibble(coefficient_table)
      )
    }

    E1_trial_rating_row_id <- E1_trial_rating_row_id + 1

    write.csv(
      bind_rows(E1_trial_rating_results),
      E1_trial_rating_output_file,
      row.names = FALSE
    )

    cat(
      sprintf(
        "Completed E1 trial-wise rating model %d/18: %s (%s)\n",
        E1_trial_rating_row_id - 1,
        outcome,
        model_type
      )
    )
  }
}

cat("\nE1 trial-wise semantic rating analysis complete.\n")
cat(
  "Saved coefficient and model-detail table to ",
  E1_trial_rating_output_file,
  "\n",
  sep = ""
)
