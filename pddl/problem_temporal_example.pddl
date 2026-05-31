;; ============================================================
;; TEMPORAL PROBLEM EXAMPLE — target format for OPTIC
;; ============================================================
;;
;; The problem file changes minimally compared to the classical version:
;;   - :init   is identical (classical predicates; no numeric fluents to initialise
;;             unless (:functions ...) is declared in the domain)
;;   - :goal   is identical
;;   - :metric is new -> tells OPTIC to minimise total plan time
;;
;; Note: (total-time) is an implicit fluent in OPTIC; it does not need to be
;; declared in the domain. It does require :numeric-fluents in the requirements.
;; ============================================================

(define (problem process_problem_prediction)

  (:domain process_domain)

  (:init
    (enabled register_er)     ;; only start activity enabled at the beginning
  )

  (:goal
    (and
      (completed discharge))  ;; end activity of the process
  )

  (:metric minimize (total-time))

)
