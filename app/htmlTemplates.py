css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://scontent.xx.fbcdn.net/v/t1.15752-9/386877621_338291365488628_4137732317643459439_n.png?stp=dst-png_p206x206&_nc_cat=103&ccb=1-7&_nc_sid=510075&_nc_ohc=fBBq8Uh11TcAX-rAYY2&_nc_ad=z-m&_nc_cid=0&_nc_ht=scontent.xx&oh=03_AdSW2za5SiDIb3k6wtbGRfJpaVb7ZMR7NWQ5UU4-LkDhPA&oe=6599D745" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''