;; ============================================================
;; TEMPORAL DOMAIN EXAMPLE — target format for OPTIC
;; ============================================================
;;
;; Exemplary process: Register -> Triage -> CRP -> Treatment/Discharge
;;
;; Durative-action structure:
;;   :duration  -> [min, max] interval derived from the log (mean +/- std, or observed min/max)
;;   :condition -> temporal preconditions:
;;       (at start ...)  conditions that must hold when the action starts
;;       (over all ...)  conditions that must hold throughout execution (shared resources, mutexes)
;;   :effect    -> temporal effects:
;;       (at start ...)  immediate effects on start (typically: disable the action)
;;       (at end   ...)  effects on completion (completed, enable successors, set attributes)
;;
;; Design rules adopted:
;;   - (enabled ?a) is removed at start -> prevents concurrent executions of the same action
;;   - (completed ?a) is added at end   -> successors can only start after this action finishes
;;   - explicit predecessor checks (completed pred) go at start
;;   - attributes are set at end; old values are removed with explicit negative effects (no forall/when)
;;
;; Note on :numeric-fluents: required for (total-time) in the metric.
;; ============================================================

(define (domain process_domain)

  (:requirements :typing :durative-actions :numeric-fluents :negative-preconditions)

  (:types
    activity
    crp_type       ;; type for the discretised values of the CRP attribute
  )

  (:constants
    register_er triage crp administer_antibiotic discharge - activity
    low medium high - crp_type
  )

  (:predicates
    (completed ?a - activity)
    (enabled   ?a - activity)
    (crp_val   ?v - crp_type)   ;; CRP attribute: at most one value true at a time
  )

  ;; ==========================================================
  ;; SIMPLE ACTION -- no attribute dependencies
  ;; ==========================================================

  (:durative-action exec_register_er
    :parameters ()
    :duration (and (>= ?duration 3.0) (<= ?duration 15.0))
    :condition (at start
      (enabled register_er))
    :effect (and
      (at start (not (enabled register_er)))
      (at end   (completed register_er))
      (at end   (enabled triage)))
  )

  ;; ----------

  (:durative-action exec_triage
    :parameters ()
    :duration (and (>= ?duration 10.0) (<= ?duration 30.0))
    :condition (and
      (at start (enabled triage))
      (at start (completed register_er)))   ;; explicit predecessor check
    :effect (and
      (at start (not (enabled triage)))
      (at end   (completed triage))
      (at end   (enabled crp)))
  )

  ;; ==========================================================
  ;; MEASUREMENT ACTION -- one variant per possible outcome
  ;;
  ;; Old attribute values are cleared with explicit enumerated
  ;; negative effects (not (crp_val X)) instead of the forall/when
  ;; pattern -> compatible with both OPTIC and Fast-Downward.
  ;; ==========================================================

  (:durative-action exec_crp-low
    :parameters ()
    :duration (and (>= ?duration 20.0) (<= ?duration 60.0))
    :condition (and
      (at start (enabled crp))
      (at start (completed triage)))
    :effect (and
      (at start (not (enabled crp)))
      (at end   (completed crp))
      (at end   (not (crp_val medium)))     ;; explicit clear of other values
      (at end   (not (crp_val high)))
      (at end   (crp_val low))
      (at end   (enabled discharge)))       ;; low CRP -> goes directly to discharge
  )

  (:durative-action exec_crp-medium
    :parameters ()
    :duration (and (>= ?duration 20.0) (<= ?duration 60.0))
    :condition (and
      (at start (enabled crp))
      (at start (completed triage)))
    :effect (and
      (at start (not (enabled crp)))
      (at end   (completed crp))
      (at end   (not (crp_val low)))
      (at end   (not (crp_val high)))
      (at end   (crp_val medium))
      (at end   (enabled administer_antibiotic)))
  )

  (:durative-action exec_crp-high
    :parameters ()
    :duration (and (>= ?duration 20.0) (<= ?duration 60.0))
    :condition (and
      (at start (enabled crp))
      (at start (completed triage)))
    :effect (and
      (at start (not (enabled crp)))
      (at end   (completed crp))
      (at end   (not (crp_val low)))
      (at end   (not (crp_val medium)))
      (at end   (crp_val high))
      (at end   (enabled administer_antibiotic)))
  )

  ;; ==========================================================
  ;; DECISION ACTION -- one variant per path from decision mining
  ;;
  ;; The guard (crp_val X) at start is the condition extracted by
  ;; the decision tree. The fallback variant covers uncovered paths.
  ;; ==========================================================

  (:durative-action exec_administer_antibiotic_v1
    :parameters ()
    :duration (and (>= ?duration 30.0) (<= ?duration 120.0))
    :condition (and
      (at start (enabled administer_antibiotic))
      (at start (completed crp))
      (at start (crp_val high)))            ;; decision mining guard
    :effect (and
      (at start (not (enabled administer_antibiotic)))
      (at end   (completed administer_antibiotic))
      (at end   (enabled discharge)))
  )

  (:durative-action exec_administer_antibiotic_v2
    :parameters ()
    :duration (and (>= ?duration 30.0) (<= ?duration 120.0))
    :condition (and
      (at start (enabled administer_antibiotic))
      (at start (completed crp))
      (at start (crp_val medium)))          ;; decision mining guard
    :effect (and
      (at start (not (enabled administer_antibiotic)))
      (at end   (completed administer_antibiotic))
      (at end   (enabled discharge)))
  )

  ;; fallback: no attribute guard (decision mining did not cover this path)
  (:durative-action exec_administer_antibiotic_fallback
    :parameters ()
    :duration (and (>= ?duration 30.0) (<= ?duration 120.0))
    :condition (and
      (at start (enabled administer_antibiotic))
      (at start (completed crp)))
    :effect (and
      (at start (not (enabled administer_antibiotic)))
      (at end   (completed administer_antibiotic))
      (at end   (enabled discharge)))
  )

  ;; ==========================================================
  ;; END ACTIVITY
  ;; Reachable from multiple predecessors (crp-low or administer_antibiotic):
  ;; only (enabled discharge) is checked, not a specific predecessor.
  ;; ==========================================================

  (:durative-action exec_discharge
    :parameters ()
    :duration (and (>= ?duration 5.0) (<= ?duration 20.0))
    :condition (at start
      (enabled discharge))
    :effect (and
      (at start (not (enabled discharge)))
      (at end   (completed discharge)))
  )

)
