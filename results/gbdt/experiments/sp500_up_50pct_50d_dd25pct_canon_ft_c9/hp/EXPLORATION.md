# sp500 +50%/50d canonical fine-tune — HP one-by-one (select on EVAL)
FS locked candidate: 144 features (r03) — best eval AUC 0.9038 / val AUC 0.9537 / val R-p@3 0.228 / eval R-p@10 0.399.
Feature-count @ mcw10: 26=evAUC.888/rp3.253 | 60=.900/.296 | 116=.901/.306 | 144=.904/.285
HP tried (all depth6/ss.85/cs1.0/g0/eta.05 unless noted):
- 60f mcw1  (val-selected): eval rp@3 .242 @10 .332 ; eval AUC .914
- 60f mcw10 (c1): eval rp@3 .296 @10 .367 ; eval AUC .900
- 60f mcw10 cs.7 (c2): eval rp@3 .286 @10 .386  (colsample redundant w/ mcw)
- 26f mcw10 (c3): eval rp@3 .253 ; 116f mcw10 (c4): .306 ; 144f mcw10 (c5): .285 @10 .399
Current best-supported: 144f mcw10. Next: mcw magnitude {20}, then max_depth.
