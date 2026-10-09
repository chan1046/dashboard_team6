from sqlalchemy import create_engine, String, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column,Session
 
class Base(DeclarativeBase):  #()안에 있는 것을 상속받아 사용
    pass
class User(Base):
    __tablename__ = "users"

    id = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    name = mapped_column(String(50))
    email = mapped_column(String(100), unique=True)
DATABASE_URL = "mysql_pymysql://root:0000@localhost:3306/testdb"

engine = create_engine(
    DATABASE_URL,
    echo=True,
)

Base.metadata.create_all(engine)

with Session(engine) as session:
    user = User(
        name="홍길동",
        email="hong@example.com"
    )

    session.add(user)
    session.commit()

    print("생성된 ID:", user.id)