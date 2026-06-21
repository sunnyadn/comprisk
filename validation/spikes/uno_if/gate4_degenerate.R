suppressMessages(library(survC1))
d <- read.csv("/tmp/gate4_data.csv")
mydata <- data.frame(time = d$time, status = d$status, score = d$score)
tau <- max(d$time) * 1.01  # no truncation

# point estimate with the FIXED score (nofit) — should equal crforest exactly
pt <- Est.Cval(mydata, tau, nofit = TRUE)$Dhat

# inference: Inf.Cval fits a Cox on the single covariate; because C is rank-based
# and a single positive coefficient preserves ranks, the model (Wb) term ~ 0, so
# the SE here == Wa + Wg == crforest's fixed-prediction IF SE.
set.seed(1)
inf <- Inf.Cval(mydata, tau, itr = 500)

cat(sprintf("[gate4 survC1 ] n=%d\n", nrow(mydata)))
cat(sprintf("  Est.Cval (nofit, fixed score) = %.6f\n", pt))
cat(sprintf("  Inf.Cval Dhat                 = %.6f\n", inf$Dhat))
cat(sprintf("  Inf.Cval se (perturbation)    = %.6f\n", inf$se))
