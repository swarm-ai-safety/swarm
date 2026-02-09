"""Shared enums/specs for simulated API tasks."""

from __future__ import annotations

from enum import Enum


class Domain(str, Enum):
    IAM = "iam"
    PAYMENTS = "payments"
    INCIDENT_RESPONSE = "incident_response"


class Split(str, Enum):
    TUNING = "tuning"
    HELD_OUT = "held_out"
