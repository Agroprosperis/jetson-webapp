Start the app first, then run:

```bash
cd app
TILLETIA_ADMIN_PASSWORD='<admin password>' python3 -m unittest discover -s tests
```

The tests assume:
- local auth exists
- bearer access tokens are used
- the `admin` password was changed from the default `admin` and is passed via `TILLETIA_ADMIN_PASSWORD`
