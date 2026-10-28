"""Save this computer's League client location during installation, offline."""

from services.draft_import import configure_client_location


def main():
    path, saved = configure_client_location()
    print(f"[LoL] Lockfile helye: {path}")
    if saved:
        print("[LoL] Az utvonal elmentve ezen a gepen. A kliens indulasa utan automatikusan hasznalhato.")
    else:
        print("[LoL] Az utvonal most nem mentheto; az alkalmazas ujra megprobalja az elso beolvasaskor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
