from elasticsearch_dsl.connections import connections
from elasticsearch import Elasticsearch


class EsClient:
    def __init__(
        self, host: str = "localhost", port: int = 9200, protocol: str = "https"
    ):
        self.host = host
        self.port = port
        self.url = f"{protocol}://{self.host}:{self.port}"
        self.is_connected = False

    def connect(self) -> Elasticsearch:
        if self.is_connected:
            return
        client = connections.create_connection(
            hosts=[self.url],
            basic_auth=("elastic", "wqS47Q3vYra52JNzPkwJ"),
            verify_certs=False,
        )
        return client

    def get_all_index(self) -> list[str]:
        return list(connections.get_connection().indices.get_alias().keys())


def es_connection() -> Elasticsearch:
    cli = EsClient()
    return cli.connect()


if __name__ == "__main__":
    cli = EsClient()
    client = cli.connect()
    print(type(client))
    print(cli.get_all_index())
