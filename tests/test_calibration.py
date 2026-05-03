from app.calibration import calibrate_confidence


def test_calibration_basic():
    sample = [
        {
            "stance": "supports",
            "confidence": 0.9,
            "paper_id": "A",
            "citation_count": 50,
            "year": 2020,
            "score": 0.8,
        },
        {
            "stance": "contradicts",
            "confidence": 0.85,
            "paper_id": "B",
            "citation_count": 40,
            "year": 2019,
            "score": 0.75,
        },
        {
            "stance": "neutral",
            "confidence": 0.7,
            "paper_id": "C",
            "citation_count": 10,
            "year": 2021,
            "score": 0.6,
        },
    ]

    result = calibrate_confidence(sample)

    print("\n=== CALIBRATION RESULT ===\n")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    test_calibration_basic()
